"""RacetrackJaxEnv — pure-functional JAX racetrack environment.

The ego vehicle follows a curved track defined by waypoints.
Traffic vehicles are scattered on the track.

Observation: OccupancyGrid (N_GRID_FEATURES * N_GRID * N_GRID,) = (2*12*12,) = 288
  Features per cell: [presence, on_road]
  Grid: 12×12 cells covering [-18, 18] × [-18, 18] meters around ego,
        aligned to vehicle heading axes.
Action: (2,) — but only steering is used; acceleration = 0 (constant speed).
         The gym env uses lateral-only action (1D), but pearl_jax auto-pads to 2D.
"""

from __future__ import annotations

from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import jax.random as jrandom

from .state import (
    VehicleState,
    RacetrackEnvState,
    RacetrackEnvParams,
    N_MAX,
    N_GRID,
    N_GRID_FEATURES,
    N_TRACK_WAYPOINTS,
    N_RACETRACK_SIM_PER_POLICY,
)
from .kinematics import update_vehicles_kinematics
from .collision import check_ego_collision
from .utils import lmap


# ======================================================================
#  Track definition — an oval racetrack as waypoints
# ======================================================================

def _build_oval_track(
    n_points: int = N_TRACK_WAYPOINTS,
    straight_length: float = 80.0,
    radius: float = 30.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build an oval track as (x, y) waypoint arrays.

    The track consists of two straight sections connected by semicircles.
    Returns arrays of shape (N_TRACK_WAYPOINTS,).
    """
    # Parameterize by arc length fraction
    half_straight = straight_length / 2.0
    semicircle_len = jnp.pi * radius
    total_len = 2 * straight_length + 2 * semicircle_len

    t = jnp.linspace(0, total_len, n_points, endpoint=False)

    def _point(s):
        """Map arc-length s to (x, y) on the oval."""
        # Segment 1: bottom straight (left to right)
        s1 = straight_length
        # Segment 2: right semicircle
        s2 = s1 + semicircle_len
        # Segment 3: top straight (right to left)
        s3 = s2 + straight_length
        # Segment 4: left semicircle
        # s4 = total

        # Bottom straight: y = -radius, x from -half to +half
        x1 = -half_straight + s
        y1 = -radius
        in_seg1 = s < s1

        # Right semicircle: center at (half_straight, 0)
        angle2 = -jnp.pi / 2 + (s - s1) / radius
        x2 = half_straight + radius * jnp.cos(angle2)
        y2 = radius * jnp.sin(angle2)
        in_seg2 = (s >= s1) & (s < s2)

        # Top straight: y = radius, x from +half to -half
        x3 = half_straight - (s - s2)
        y3 = radius
        in_seg3 = (s >= s2) & (s < s3)

        # Left semicircle: center at (-half_straight, 0)
        angle4 = jnp.pi / 2 + (s - s3) / radius
        x4 = -half_straight + radius * jnp.cos(angle4)
        y4 = radius * jnp.sin(angle4)
        # in_seg4 = s >= s3

        x = jnp.where(in_seg1, x1,
            jnp.where(in_seg2, x2,
            jnp.where(in_seg3, x3, x4)))
        y = jnp.where(in_seg1, y1,
            jnp.where(in_seg2, y2,
            jnp.where(in_seg3, y3, y4)))
        return x, y

    xs, ys = jax.vmap(_point)(t)
    return xs, ys


# Pre-compute track waypoints (compile-time constant)
_TRACK_X, _TRACK_Y = _build_oval_track()


def _nearest_track_point(
    px: jnp.ndarray,
    py: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find nearest track waypoint to a point.

    Returns:
        (distance, nearest_x, nearest_y)
    """
    dx = _TRACK_X - px
    dy = _TRACK_Y - py
    dist_sq = dx * dx + dy * dy
    idx = jnp.argmin(dist_sq)
    return jnp.sqrt(dist_sq[idx]), _TRACK_X[idx], _TRACK_Y[idx]


def _is_on_road(px: jnp.ndarray, py: jnp.ndarray, road_width: float) -> jnp.ndarray:
    """Check if a point is on the road (within road_width/2 of track center)."""
    dist, _, _ = _nearest_track_point(px, py)
    return dist < road_width / 2.0


class RacetrackJaxEnv:
    """Pure-functional JAX racetrack environment."""

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def reset(
        key: jnp.ndarray,
        params: RacetrackEnvParams,
    ) -> Tuple[jnp.ndarray, RacetrackEnvState]:
        """Initialize a fresh racetrack episode."""
        key_ego, key_traffic, key_speed = jrandom.split(key, 3)

        n_vehicles = jnp.minimum(params.num_vehicles, N_MAX)

        # --- Ego (index 0): start at first track waypoint ---
        ego_x = _TRACK_X[0]
        ego_y = _TRACK_Y[0]
        # Heading: direction toward next waypoint
        dx = _TRACK_X[1] - _TRACK_X[0]
        dy = _TRACK_Y[1] - _TRACK_Y[0]
        ego_heading = jnp.arctan2(dy, dx)
        ego_speed = params.idm_desired_speed

        # --- Traffic: place at random track positions ---
        traffic_indices = jrandom.randint(
            key_traffic, (N_MAX - 1,),
            minval=N_TRACK_WAYPOINTS // 4,
            maxval=N_TRACK_WAYPOINTS * 3 // 4,
        )
        traffic_x = _TRACK_X[traffic_indices]
        traffic_y = _TRACK_Y[traffic_indices]

        # Heading: toward next waypoint
        next_indices = (traffic_indices + 1) % N_TRACK_WAYPOINTS
        tdx = _TRACK_X[next_indices] - traffic_x
        tdy = _TRACK_Y[next_indices] - traffic_y
        traffic_heading = jnp.arctan2(tdy, tdx)

        speed_noise = jrandom.uniform(key_speed, (N_MAX - 1,), minval=-2.0, maxval=2.0)
        traffic_speed = jnp.clip(
            params.idm_desired_speed + speed_noise, 2.0, params.max_speed,
        )

        # Assemble
        # NOTE: kinematic bicycle model treats `vx` as the scalar SPEED
        # magnitude (along heading), not the x-velocity component.  Storing
        # the x-component here would give negative speeds for vehicles whose
        # heading has cos<0 (e.g. on the top straight or left semicircle),
        # which causes them to stop or reverse.  We must store the magnitude.
        x = jnp.concatenate([jnp.array([ego_x]), traffic_x])
        y = jnp.concatenate([jnp.array([ego_y]), traffic_y])
        heading = jnp.concatenate([jnp.array([ego_heading]), traffic_heading])
        vx = jnp.concatenate([jnp.array([ego_speed]), traffic_speed])
        vy = jnp.zeros(N_MAX)

        active = jnp.arange(N_MAX) < n_vehicles
        lane_id = jnp.zeros(N_MAX, dtype=jnp.int32)
        target_lane_id = jnp.zeros(N_MAX, dtype=jnp.int32)
        length = jnp.full(N_MAX, params.vehicle_length)
        width = jnp.full(N_MAX, params.vehicle_width)
        target_speed = jnp.concatenate([
            jnp.array([ego_speed]),
            traffic_speed,
        ])

        vehicles = VehicleState(
            x=x, y=y, vx=vx, vy=vy,
            heading=heading,
            active=active,
            lane_id=lane_id,
            target_lane_id=target_lane_id,
            length=length,
            width=width,
            target_speed=target_speed,
        )

        state = RacetrackEnvState(
            vehicles=vehicles,
            time=jnp.array(0.0),
            step_count=jnp.array(0, dtype=jnp.int32),
            crashed=jnp.array(False),
            truncated=jnp.array(False),
            done=jnp.array(False),
            off_road=jnp.array(False),
        )

        obs = _observe_racetrack(state, params)
        return obs, state

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def step(
        key: jnp.ndarray,
        state: RacetrackEnvState,
        action: jnp.ndarray,
        params: RacetrackEnvParams,
    ) -> Tuple[jnp.ndarray, RacetrackEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """One policy step on the racetrack.

        Action: [acceleration, steering]. In the gym env, only steering is
        used (longitudinal=False), but we accept 2D for compatibility.
        """
        # Use only steering; maintain constant speed (like gym racetrack)
        ego_accel = jnp.array(0.0)  # ignore action[0] entirely
        # Clip steering to gym's range: [-deg2rad(45), deg2rad(45)] ≈ [-0.785, 0.785]
        _STEERING_RANGE = jnp.pi / 4.0  # 45 degrees
        ego_steering = jnp.clip(action[-1], -_STEERING_RANGE, _STEERING_RANGE)

        def _sim_step(carry, _):
            veh, crashed, sim_key = carry
            sim_key, _ = jrandom.split(sim_key)

            # Traffic: constant speed, follow track heading
            # Steer traffic toward nearest track waypoint
            traffic_steerings = _track_following_steering(veh, params)
            traffic_accels = jnp.zeros(N_MAX)

            accels = traffic_accels.at[0].set(ego_accel)
            steerings = traffic_steerings.at[0].set(ego_steering)

            veh = update_vehicles_kinematics(
                veh, accels, steerings, params.dt, params.max_speed,
            )

            ego_crashed = check_ego_collision(veh)
            crashed = crashed | ego_crashed

            return (veh, crashed, sim_key), None

        (vehicles, crashed, _), _ = jax.lax.scan(
            _sim_step,
            (state.vehicles, state.crashed, key),
            None,
            length=N_RACETRACK_SIM_PER_POLICY,
        )

        # Check if ego is off road
        ego_off_road = ~_is_on_road(vehicles.x[0], vehicles.y[0], params.road_width)

        new_time = state.time + 1.0
        new_step = state.step_count + 1
        truncated = new_step >= params.max_steps
        done = crashed | truncated | ego_off_road

        new_state = RacetrackEnvState(
            vehicles=vehicles,
            time=new_time,
            step_count=new_step,
            crashed=crashed,
            truncated=truncated,
            done=done,
            off_road=ego_off_road,
        )

        obs = _observe_racetrack(new_state, params)
        reward = _compute_racetrack_reward(new_state, action, params)

        info = {
            "crashed": crashed,
            "truncated": truncated,
            "off_road": ego_off_road,
            "step_count": new_step,
        }

        return obs, new_state, reward, done, info

    # ------------------------------------------------------------------
    #  Auto-reset step
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def step_auto_reset(
        key: jnp.ndarray,
        state: RacetrackEnvState,
        action: jnp.ndarray,
        params: RacetrackEnvParams,
    ) -> Tuple[jnp.ndarray, RacetrackEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Step with auto-reset on done."""
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = RacetrackJaxEnv.step(
            key_step, state, action, params,
        )
        obs_reset, state_reset = RacetrackJaxEnv.reset(key_reset, params)

        next_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(done, a, b),
            state_reset, next_state,
        )
        obs = jnp.where(done, obs_reset, obs)

        return obs, next_state, reward, done, info


# ======================================================================
#  Track-following steering for traffic
# ======================================================================

def _track_following_steering(
    vehicles: VehicleState,
    params: RacetrackEnvParams,
    kp: float = 0.5,
) -> jnp.ndarray:
    """Compute steering for all vehicles to follow the track.

    Each vehicle steers toward the nearest track waypoint ahead.
    """
    # For each vehicle, find nearest track waypoint
    # Then steer toward the next waypoint
    def _steer_one(vx, vy, vh, speed):
        # Find nearest waypoint
        dx = _TRACK_X - vx
        dy = _TRACK_Y - vy
        dist_sq = dx * dx + dy * dy
        nearest_idx = jnp.argmin(dist_sq)

        # Target: a few waypoints ahead
        target_idx = (nearest_idx + 3) % N_TRACK_WAYPOINTS
        target_x = _TRACK_X[target_idx]
        target_y = _TRACK_Y[target_idx]

        # Desired heading
        desired_heading = jnp.arctan2(target_y - vy, target_x - vx)
        heading_error = desired_heading - vh
        # Wrap to [-pi, pi]
        heading_error = jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))

        steering = kp * heading_error
        return jnp.clip(steering, -jnp.pi / 4.0, jnp.pi / 4.0)

    steerings = jax.vmap(_steer_one)(
        vehicles.x, vehicles.y, vehicles.heading,
        jnp.sqrt(vehicles.vx ** 2 + vehicles.vy ** 2),
    )

    # Ego steering is overridden by the policy, so zero it here
    steerings = steerings.at[0].set(0.0)
    # Zero inactive vehicles
    steerings = jnp.where(vehicles.active, steerings, 0.0)

    return steerings


# ======================================================================
#  OccupancyGrid observation
# ======================================================================

# Pre-compute grid cell centers (12×12)
_GRID_OFFSETS = jnp.arange(N_GRID) * 3.0 + 1.5 - 18.0  # [-16.5, -13.5, ..., 16.5]


def _observe_racetrack(
    state: RacetrackEnvState,
    params: RacetrackEnvParams,
) -> jnp.ndarray:
    """Compute OccupancyGrid observation.

    Grid is centered on ego vehicle, aligned to ego heading.
    Each cell has 2 features: [presence, on_road].

    Returns:
        Flat vector of shape (N_GRID_FEATURES * N_GRID * N_GRID,) = (288,).
    """
    vehicles = state.vehicles
    ego_x = vehicles.x[0]
    ego_y = vehicles.y[0]
    ego_h = vehicles.heading[0]

    cos_h = jnp.cos(ego_h)
    sin_h = jnp.sin(ego_h)

    # Grid cell centers in ego frame
    gx, gy = jnp.meshgrid(_GRID_OFFSETS, _GRID_OFFSETS)  # (12, 12)

    # Transform to world frame
    world_x = ego_x + gx * cos_h - gy * sin_h
    world_y = ego_y + gx * sin_h + gy * cos_h

    # Feature 1: on_road — is each cell on the track?
    def _check_on_road(wx, wy):
        dx = _TRACK_X - wx
        dy = _TRACK_Y - wy
        min_dist = jnp.sqrt(jnp.min(dx * dx + dy * dy))
        return (min_dist < params.road_width / 2.0).astype(jnp.float32)

    on_road = jax.vmap(jax.vmap(_check_on_road))(world_x, world_y)  # (12, 12)

    # Feature 2: presence — is there a vehicle in each cell?
    # Check if any active non-ego vehicle center falls within each cell
    cell_half = params.grid_step / 2.0

    def _check_presence_one_cell(wx, wy):
        # Distance from each vehicle to this cell center, in ego frame
        veh_dx = vehicles.x - wx
        veh_dy = vehicles.y - wy
        # Rotate to ego frame
        local_x = veh_dx * cos_h + veh_dy * sin_h
        local_y = -veh_dx * sin_h + veh_dy * cos_h
        in_cell = (jnp.abs(local_x) < cell_half) & (jnp.abs(local_y) < cell_half)
        in_cell = in_cell & vehicles.active
        # Exclude ego
        in_cell = in_cell.at[0].set(False)
        return jnp.any(in_cell).astype(jnp.float32)

    presence = jax.vmap(jax.vmap(_check_presence_one_cell))(world_x, world_y)  # (12, 12)

    # Stack: (2, 12, 12) → flatten to (288,)
    grid = jnp.stack([presence, on_road], axis=0)  # (2, 12, 12)
    return grid.flatten()


# ======================================================================
#  Reward
# ======================================================================

def _compute_racetrack_reward(
    state: RacetrackEnvState,
    action: jnp.ndarray,
    params: RacetrackEnvParams,
) -> jnp.ndarray:
    """Compute racetrack reward.

    Components:
      - collision penalty
      - lane centering reward (closer to track center = better)
      - action penalty (large steering = bad)
    """
    crashed = state.crashed.astype(jnp.float32)

    # Lane centering: distance to nearest track point
    dist_to_center, _, _ = _nearest_track_point(
        state.vehicles.x[0], state.vehicles.y[0],
    )
    # Lateral distance (meters) — matches Gym's lane.local_coordinates lateral
    lateral = dist_to_center

    # Centering reward: inverse quadratic, matching Gym's 1/(1 + cost * lateral^2)
    centering = 1.0 / (1.0 + params.lane_centering_cost * lateral ** 2)

    # FIX S1-4: action[0] (acceleration) is zeroed in dynamics (ego_accel = 0.0),
    # so penalizing it in the reward creates a misleading gradient — the policy is
    # punished for an action dimension that has no effect on the environment.
    # Gym racetrack uses lateral-only action (1D), so action_norm = |steering|.
    # We match that: compute norm over steering component only (action[-1]).
    action_norm = jnp.abs(action[-1])

    # On-road check
    is_on_road = (1.0 - state.off_road.astype(jnp.float32))

    reward = (
        params.collision_reward * crashed
        + params.lane_centering_reward * centering
        + params.action_reward * action_norm
    )

    # Normalize to [0, 1] matching Gym: lmap([collision, 1], [0, 1])
    reward = lmap(reward, (params.collision_reward, 1.0), (0.0, 1.0))

    # On-road multiplicative gate (match Gym behavior)
    reward = reward * is_on_road

    return reward
