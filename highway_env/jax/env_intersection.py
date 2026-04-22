"""IntersectionJaxEnv — pure-functional JAX 4-way intersection environment.

Vehicles approach from 4 directions on straight roads that cross at the
origin.  The ego vehicle starts from one direction and must safely cross.

Observation: (N_OBS_INTERSECTION * N_FEATURES_INTERSECTION,) = (15*7,) = 105
  Features: [presence, x, y, vx, vy, cos_h, sin_h]  — absolute, normalized.
Action: (2,) — [acceleration, steering].
"""

from __future__ import annotations

from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import jax.random as jrandom

from .state import (
    VehicleState,
    IntersectionEnvState,
    IntersectionEnvParams,
    N_MAX,
    N_OBS_INTERSECTION,
    N_FEATURES_INTERSECTION,
    N_SIM_PER_POLICY,
)
from .kinematics import update_vehicles_kinematics
from .collision import check_ego_collision
from .utils import lmap
from .discrete_action import (
    apply_meta_action_longi,
    compute_ego_control_intersection,
    INTERSECTION_TARGET_SPEEDS,
    N_DISCRETE_ACTIONS_LONGI,
)


# RESOLVED S1-3: Intersection traffic with IDM + priority/yield logic.
#
# Gym intersection_env.py:319-382 uses full IDMVehicle with yielding + route planning
# on a circular-lane road network with per-lane priority levels (3/2/1/0).
#
# INTENTIONAL_SIMPLIFICATION (B안): JAX constraint — no Python objects, fixed array
# sizes, pure functions. We implement:
#   (a) Conflict zone detection: vehicles within intersection_size radius of origin
#   (b) First-come-first-served priority: vehicle closer to conflict zone center wins
#   (c) Yield = decelerate to near-zero until the conflict zone clears
#
# This qualitatively matches gym's yielding behavior without requiring dynamic
# routing or per-lane priority tables in a JIT context.


def _intersection_priority_yield(
    vehicles: VehicleState,
    params: IntersectionEnvParams,
) -> jnp.ndarray:
    """Compute yield deceleration adjustments for intersection priority logic.

    RESOLVED S1-3 (B안 — INTENTIONAL_SIMPLIFICATION):
    Implements first-come-first-served conflict zone priority:
      - A "conflict zone" is the intersection area (radius = intersection_size from origin).
      - For each vehicle i approaching the conflict zone, check all other vehicles j.
      - Vehicle i must yield to vehicle j if:
          * Both are approaching the conflict zone (not yet in it), AND
          * Vehicle j is closer to the conflict zone center (lower time-to-conflict), AND
          * Vehicle j is on a different approach direction (crossing path).
      - Yield action: decelerate at -idm_comfort_decel until the zone is clear.

    Gym reference: behavior.py IDMVehicle + RegulatedRoad priority lanes.
    Qualitative match: lower-priority vehicle slows/stops before intersection.

    Returns:
        yield_accels: (N_MAX,) yield deceleration overrides.
                      0.0 where no yield is needed; negative where yield applies.
                      Index 0 (ego) always returns 0.0 (ego is policy-controlled).
    """
    # Distance from origin (conflict zone center)
    dist_to_center = jnp.sqrt(vehicles.x ** 2 + vehicles.y ** 2)

    # "Approaching" = active, not yet inside conflict zone, moving toward center
    # (dot product of velocity with direction toward center is positive)
    toward_center_x = -vehicles.x / jnp.maximum(dist_to_center, 1e-6)
    toward_center_y = -vehicles.y / jnp.maximum(dist_to_center, 1e-6)
    speed_mag = jnp.maximum(vehicles.vx, 0.0)
    vel_x = speed_mag * jnp.cos(vehicles.heading)
    vel_y = speed_mag * jnp.sin(vehicles.heading)
    dot_toward = vel_x * toward_center_x + vel_y * toward_center_y

    in_conflict_zone = dist_to_center < params.intersection_size
    approaching = vehicles.active & (~in_conflict_zone) & (dot_toward > 0.5)

    # Time-to-conflict: how long until each approaching vehicle reaches the conflict zone
    # Approximate: (dist_to_center - intersection_size) / speed
    approach_dist = jnp.maximum(dist_to_center - params.intersection_size, 0.0)
    ttc = approach_dist / jnp.maximum(speed_mag, 0.5)  # seconds

    # Approach direction: quantized into 4 arms based on heading
    # Direction 0 (S→N): heading ~ pi/2,  Direction 1 (W→E): heading ~ 0
    # Direction 2 (N→S): heading ~ -pi/2, Direction 3 (E→W): heading ~ pi
    # Use cos/sin of heading to classify: N/S vs E/W
    cos_h = jnp.cos(vehicles.heading)
    sin_h = jnp.sin(vehicles.heading)
    # NS arm: |sin_h| > |cos_h|;  EW arm: |cos_h| > |sin_h|
    arm_ns = jnp.abs(sin_h) > jnp.abs(cos_h)  # True = N or S arm, False = E or W arm

    def _yield_for_vehicle(i):
        """Determine if vehicle i should yield (returns yield decel magnitude >= 0)."""
        i_approaching = approaching[i]
        i_ttc = ttc[i]
        i_arm_ns = arm_ns[i]

        # Check all other vehicles j: does i need to yield to j?
        # Yield condition: i and j on different arms, j is closer (lower ttc), j approaching
        diff_arm = (arm_ns != i_arm_ns)
        j_has_priority = approaching & diff_arm & (ttc < i_ttc - 0.5)
        # Also yield if another vehicle is already in the conflict zone on a crossing arm
        j_in_zone_crossing = vehicles.active & in_conflict_zone & diff_arm

        must_yield = jnp.any(j_has_priority | j_in_zone_crossing)

        # Yield deceleration: -idm_comfort_decel when must yield and approaching
        yield_decel = jnp.where(
            i_approaching & must_yield,
            -params.idm_comfort_decel,
            0.0,
        )
        # Ego (index 0) is always policy-controlled — no yield override
        yield_decel = jnp.where(i == 0, 0.0, yield_decel)
        return yield_decel

    yield_accels = jax.vmap(_yield_for_vehicle)(jnp.arange(N_MAX))
    return yield_accels


def _intersection_traffic_idm(
    vehicles: VehicleState,
    params: IntersectionEnvParams,
) -> jnp.ndarray:
    """Compute IDM + priority/yield accelerations for intersection traffic.

    RESOLVED S1-3: Combines two layers:
      1. Base IDM: decelerate when a vehicle is ahead along heading direction.
      2. Priority yield override: if vehicle must yield at intersection, apply
         -idm_comfort_decel regardless of IDM (takes the more negative value).

    Gym reference: IDMVehicle.act() + RegulatedRoad priority system.
    INTENTIONAL_SIMPLIFICATION: first-come-first-served instead of per-lane
    priority levels; straight-road kinematics only (no circular lane routing).

    Returns:
        accels: (N_MAX,) longitudinal accelerations. Index 0 (ego) is left as 0.0
                and will be overridden by the policy.
    """
    # Detection horizon for "ahead" vehicles
    detect_horizon = 60.0  # meters
    a_max = params.idm_comfort_accel
    v0 = params.idm_desired_speed
    min_gap = params.idm_min_gap + params.vehicle_length  # s0 = min_gap + length
    time_gap = params.idm_time_gap
    ab_sqrt = jnp.sqrt(jnp.maximum(params.idm_comfort_accel * params.idm_comfort_decel, 1e-6))

    def _idm_accel_for_vehicle(i):
        xi = vehicles.x[i]
        yi = vehicles.y[i]
        vi = vehicles.vx[i]  # scalar speed magnitude
        hi = vehicles.heading[i]
        cos_hi = jnp.cos(hi)
        sin_hi = jnp.sin(hi)

        # Vector from vehicle i to each other vehicle j
        dx = vehicles.x - xi
        dy = vehicles.y - yi

        # Project onto heading direction of vehicle i (longitudinal component)
        long_dist = dx * cos_hi + dy * sin_hi

        # Only vehicles ahead (long_dist > 0) and within detection horizon
        ahead = (long_dist > 0.5) & (long_dist < detect_horizon) & vehicles.active
        # Exclude self
        ahead = ahead.at[i].set(False)

        # Gap = longitudinal distance minus vehicle length
        gap = long_dist - params.vehicle_length

        # Find minimum gap among vehicles ahead
        gap_masked = jnp.where(ahead, gap, detect_horizon)
        min_idx = jnp.argmin(gap_masked)
        min_gap_val = gap_masked[min_idx]
        has_front = jnp.any(ahead)
        front_speed = vehicles.vx[min_idx]

        # IDM free-flow term
        v_ratio = jnp.maximum(vi, 0.0) / jnp.maximum(v0, 1e-6)
        accel_free = a_max * (1.0 - jnp.power(v_ratio, 4.0))

        # IDM interaction term: s* = s0 + v*T + v*(v-v_front)/(2*sqrt(a*b))
        dv = vi - front_speed
        s_star = min_gap + vi * time_gap + vi * dv / (2.0 * ab_sqrt)
        s_star = jnp.maximum(s_star, min_gap)
        gap_clamped = jnp.maximum(min_gap_val, 0.1)
        accel_follow = a_max * (-(s_star / gap_clamped) ** 2)

        # Combine: free + follow (only apply follow term if there's a front vehicle)
        accel = accel_free + jnp.where(has_front, accel_follow, 0.0)
        accel = jnp.clip(accel, -params.idm_comfort_decel, a_max)

        # Zero out for inactive vehicles
        accel = jnp.where(vehicles.active[i], accel, 0.0)
        return accel

    accels = jax.vmap(_idm_accel_for_vehicle)(jnp.arange(N_MAX))

    # RESOLVED S1-3: Apply priority/yield overrides on top of IDM.
    # Take the more negative (conservative) of IDM and yield deceleration.
    yield_accels = _intersection_priority_yield(vehicles, params)
    accels = jnp.minimum(accels, yield_accels)

    # Clip to comfort decel bounds after combining
    accels = jnp.clip(accels, -params.idm_comfort_decel, params.idm_comfort_accel)

    # Index 0 (ego) will be overridden by policy; return 0.0 for it
    accels = accels.at[0].set(0.0)
    return accels


# Direction vectors: S→N, W→E, N→S, E→W
_DIR_DX = jnp.array([0.0, 1.0, 0.0, -1.0])
_DIR_DY = jnp.array([1.0, 0.0, -1.0, 0.0])
_DIR_HEADING = jnp.array([jnp.pi / 2, 0.0, -jnp.pi / 2, jnp.pi])

# Spawn offsets: where vehicles start on each approach
# Direction 0 (S→N): x=0, y = -road_length
# Direction 1 (W→E): x=-road_length, y=0
# Direction 2 (N→S): x=0, y = road_length
# Direction 3 (E→W): x=road_length, y=0
_SPAWN_SIGN_X = jnp.array([0.0, -1.0, 0.0, 1.0])
_SPAWN_SIGN_Y = jnp.array([-1.0, 0.0, 1.0, 0.0])


class IntersectionJaxEnv:
    """Pure-functional JAX intersection environment."""

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def reset(
        key: jnp.ndarray,
        params: IntersectionEnvParams,
    ) -> Tuple[jnp.ndarray, IntersectionEnvState]:
        """Initialize a fresh intersection episode."""
        key_ego, key_dirs, key_pos, key_speed = jrandom.split(key, 4)

        # Scale active vehicle count by vehicles_density (SCM-driven)
        n_base = jnp.minimum(params.num_vehicles, N_MAX)
        n_vehicles = jnp.minimum(
            jnp.floor(n_base * params.vehicles_density).astype(jnp.int32),
            N_MAX,
        )
        n_vehicles = jnp.maximum(n_vehicles, 2)  # at least ego + 1 traffic

        # --- Ego vehicle (index 0) ---
        # Randomize ego direction: use key bits to pick approach direction
        ego_dir_key, key_ego = jrandom.split(key_ego)
        ego_dir_rand = jrandom.randint(ego_dir_key, (), 0, 4)
        # Use params.ego_direction if >= 0, else random
        ego_dir = jnp.where(params.ego_direction >= 0, params.ego_direction, ego_dir_rand)
        ego_x = _SPAWN_SIGN_X[ego_dir] * params.road_length * 0.8
        ego_y = _SPAWN_SIGN_Y[ego_dir] * params.road_length * 0.8
        ego_heading = _DIR_HEADING[ego_dir]
        # NOTE: bicycle model stores SCALAR SPEED in `vx`, not x-component.
        # The kinematic update derives world vx/vy from speed * (cos/sin)(heading + beta).
        ego_vx = params.ego_initial_speed
        ego_vy = jnp.array(0.0)

        # --- Traffic vehicles (indices 1..n_active-1) ---
        # Assign random directions to traffic
        traffic_dirs = jrandom.randint(key_dirs, (N_MAX - 1,), 0, 4)

        # Spawn positions: mix of close (dangerous) and far traffic
        # 40% spawn within 0.1-0.4 of road_length (near intersection)
        # 60% spawn within 0.4-0.9 of road_length (approaching)
        dist_raw = jrandom.uniform(key_pos, (N_MAX - 1,), minval=0.0, maxval=1.0)
        near_mask = dist_raw < 0.4
        dist_near = 0.1 + dist_raw / 0.4 * 0.3  # [0.1, 0.4]
        dist_far = 0.4 + (dist_raw - 0.4) / 0.6 * 0.5  # [0.4, 0.9]
        dist_frac = jnp.where(near_mask, dist_near, dist_far)
        dist_along = dist_frac * params.road_length

        traffic_x = _SPAWN_SIGN_X[traffic_dirs] * dist_along
        traffic_y = _SPAWN_SIGN_Y[traffic_dirs] * dist_along
        traffic_heading = _DIR_HEADING[traffic_dirs]

        # Speeds
        speed_noise = jrandom.uniform(key_speed, (N_MAX - 1,), minval=-2.0, maxval=2.0)
        traffic_speed = jnp.clip(
            params.idm_desired_speed + speed_noise, 3.0, params.max_speed
        )
        # Same convention: store scalar speed magnitude in vx; vy stays 0.
        traffic_vx = traffic_speed
        traffic_vy = jnp.zeros(N_MAX - 1)

        # --- Assemble arrays ---
        x = jnp.concatenate([jnp.array([ego_x]), traffic_x])
        y = jnp.concatenate([jnp.array([ego_y]), traffic_y])
        vx = jnp.concatenate([jnp.array([ego_vx]), traffic_vx])
        vy = jnp.concatenate([jnp.array([ego_vy]), traffic_vy])
        heading = jnp.concatenate([jnp.array([ego_heading]), traffic_heading])

        active = jnp.arange(N_MAX) < n_vehicles
        lane_id = jnp.zeros(N_MAX, dtype=jnp.int32)
        target_lane_id = jnp.zeros(N_MAX, dtype=jnp.int32)
        length = jnp.full(N_MAX, params.vehicle_length)
        width = jnp.full(N_MAX, params.vehicle_width)
        target_speed = jnp.concatenate([
            jnp.array([params.ego_initial_speed]),
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

        state = IntersectionEnvState(
            vehicles=vehicles,
            time=jnp.array(0.0),
            step_count=jnp.array(0, dtype=jnp.int32),
            crashed=jnp.array(False),
            truncated=jnp.array(False),
            done=jnp.array(False),
            arrived=jnp.array(False),
            ego_direction=jnp.array(ego_dir, dtype=jnp.int32),
        )

        obs = _observe_intersection(state, params)
        return obs, state

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def step(
        key: jnp.ndarray,
        state: IntersectionEnvState,
        action: jnp.ndarray,
        params: IntersectionEnvParams,
    ) -> Tuple[jnp.ndarray, IntersectionEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """One policy step in the intersection."""
        ego_accel = action[0]
        ego_steering = action[1]

        def _sim_step(carry, _):
            veh, crashed, sim_key = carry
            sim_key, tkey = jrandom.split(sim_key)

            # RESOLVED S1-3: IDM + priority/yield control for traffic.
            # Traffic decelerates behind lead vehicles (IDM) and yields at the
            # intersection conflict zone via first-come-first-served priority.
            # See _intersection_traffic_idm and _intersection_priority_yield.
            traffic_accels = _intersection_traffic_idm(veh, params)
            traffic_steerings = jnp.zeros(N_MAX)

            # Ego action
            accels = traffic_accels.at[0].set(ego_accel)
            steerings = traffic_steerings.at[0].set(ego_steering)

            # Kinematics
            veh = update_vehicles_kinematics(
                veh, accels, steerings, params.dt, params.max_speed,
            )

            # Collision
            ego_crashed = check_ego_collision(veh)
            crashed = crashed | ego_crashed

            return (veh, crashed, sim_key), None

        (vehicles, crashed, _), _ = jax.lax.scan(
            _sim_step,
            (state.vehicles, state.crashed, key),
            None,
            length=N_SIM_PER_POLICY,
        )

        # Check if ego arrived (crossed to the opposite side)
        # Use actual ego direction from state (may differ from params if randomized)
        ego_dir = state.ego_direction
        # Arrived when ego passes origin and reaches opposite side
        # Direction 0 (S→N): arrived when y > intersection_size
        # Direction 1 (W→E): arrived when x > intersection_size
        # Direction 2 (N→S): arrived when y < -intersection_size
        # Direction 3 (E→W): arrived when x < -intersection_size
        ego_x = vehicles.x[0]
        ego_y = vehicles.y[0]
        arrived_0 = (ego_dir == 0) & (ego_y > params.intersection_size)
        arrived_1 = (ego_dir == 1) & (ego_x > params.intersection_size)
        arrived_2 = (ego_dir == 2) & (ego_y < -params.intersection_size)
        arrived_3 = (ego_dir == 3) & (ego_x < -params.intersection_size)
        arrived = arrived_0 | arrived_1 | arrived_2 | arrived_3

        new_time = state.time + 1.0
        new_step = state.step_count + 1
        truncated = new_step >= params.max_steps
        done = crashed | truncated | arrived

        new_state = IntersectionEnvState(
            vehicles=vehicles,
            time=new_time,
            step_count=new_step,
            crashed=crashed,
            truncated=truncated,
            done=done,
            arrived=arrived,
            ego_direction=state.ego_direction,
        )

        obs = _observe_intersection(new_state, params)
        reward = _compute_intersection_reward(new_state, action, params)

        info = {
            "crashed": crashed,
            "truncated": truncated,
            "arrived": arrived,
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
        state: IntersectionEnvState,
        action: jnp.ndarray,
        params: IntersectionEnvParams,
    ) -> Tuple[jnp.ndarray, IntersectionEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Step with auto-reset on done."""
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = IntersectionJaxEnv.step(
            key_step, state, action, params,
        )

        # FIX P1-2 FINAL: use lax.cond so reset() only executes on done=True.
        # The old pattern always called reset() unconditionally. lax.cond only
        # traces and executes one branch at runtime, avoiding wasted work ~97.5% of steps.
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: IntersectionJaxEnv.reset(key_reset, params),
            false_fun=lambda _: (obs, next_state),
            operand=None,
        )

        next_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(done, a, b),
            state_reset, next_state,
        )
        obs = jnp.where(done, obs_reset, obs)

        return obs, next_state, reward, done, info

    # ------------------------------------------------------------------
    #  Discrete meta-action step (Highway-Env compatible)
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def step_discrete(
        key: jnp.ndarray,
        state: IntersectionEnvState,
        action: jnp.ndarray,
        params: IntersectionEnvParams,
    ) -> Tuple[jnp.ndarray, IntersectionEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Execute one policy step with a discrete meta-action.

        Matches Highway-Env's ``DiscreteMetaAction`` with
        ``longitudinal=True, lateral=False`` — 3 actions:
          0=SLOWER, 1=IDLE, 2=FASTER.

        Target speeds: [0, 4.5, 9] m/s (from intersection_env.py).

        Only speed is changed; heading remains fixed along the
        approach direction set at reset.

        Args:
            key: PRNG key.
            state: current environment state.
            action: scalar integer in {0, 1, 2}.
            params: intersection environment parameters.

        Returns:
            (obs, next_state, reward, done, info)
        """
        # 1. Apply longitudinal-only meta-action
        vehicles = apply_meta_action_longi(
            action, state.vehicles, INTERSECTION_TARGET_SPEEDS,
        )
        state = state.replace(vehicles=vehicles)

        # 2. Simulation loop — recompute speed control every sub-step.
        def _sim_step(carry, _):
            veh, crashed, sim_key = carry
            sim_key, tkey = jrandom.split(sim_key)

            # RESOLVED S1-3: IDM + priority/yield control for traffic (discrete step).
            # Same as continuous step: IDM following + conflict zone yield.
            traffic_accels = _intersection_traffic_idm(veh, params)
            traffic_steerings = jnp.zeros(N_MAX)

            ego_accel_mag, _ = compute_ego_control_intersection(veh, params)
            # Ego: apply acceleration along heading, no steering
            accels = traffic_accels.at[0].set(ego_accel_mag)
            steerings = traffic_steerings  # ego steering = 0

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
            length=N_SIM_PER_POLICY,
        )

        # Check arrival
        ego_dir = state.ego_direction
        ego_x = vehicles.x[0]
        ego_y = vehicles.y[0]
        arrived_0 = (ego_dir == 0) & (ego_y > params.intersection_size)
        arrived_1 = (ego_dir == 1) & (ego_x > params.intersection_size)
        arrived_2 = (ego_dir == 2) & (ego_y < -params.intersection_size)
        arrived_3 = (ego_dir == 3) & (ego_x < -params.intersection_size)
        arrived = arrived_0 | arrived_1 | arrived_2 | arrived_3

        new_time = state.time + 1.0
        new_step = state.step_count + 1
        truncated = new_step >= params.max_steps
        done = crashed | truncated | arrived

        new_state = IntersectionEnvState(
            vehicles=vehicles,
            time=new_time,
            step_count=new_step,
            crashed=crashed,
            truncated=truncated,
            done=done,
            arrived=arrived,
            ego_direction=state.ego_direction,
        )

        obs = _observe_intersection(new_state, params)
        ego_accel_final, _ = compute_ego_control_intersection(vehicles, params)
        ctrl_action = jnp.array([ego_accel_final, 0.0])
        reward = _compute_intersection_reward(new_state, ctrl_action, params)

        info = {
            "crashed": crashed,
            "truncated": truncated,
            "arrived": arrived,
            "step_count": new_step,
        }

        return obs, new_state, reward, done, info

    # ------------------------------------------------------------------
    #  Auto-reset discrete step
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def step_auto_reset_discrete(
        key: jnp.ndarray,
        state: IntersectionEnvState,
        action: jnp.ndarray,
        params: IntersectionEnvParams,
    ) -> Tuple[jnp.ndarray, IntersectionEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Discrete meta-action step with automatic reset on done."""
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = IntersectionJaxEnv.step_discrete(
            key_step, state, action, params,
        )

        # FIX P1-2 FINAL: lazy reset via lax.cond (same as step_auto_reset)
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: IntersectionJaxEnv.reset(key_reset, params),
            false_fun=lambda _: (obs, next_state),
            operand=None,
        )

        next_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(done, a, b),
            state_reset, next_state,
        )
        obs = jnp.where(done, obs_reset, obs)

        return obs, next_state, reward, done, info


# ======================================================================
#  Observation (absolute coordinates, 15 vehicles × 7 features)
# ======================================================================

# Normalisation scales: [presence, x, y, vx, vy, cos_h, sin_h]
_INTERSECTION_SCALES = jnp.array([1.0, 100.0, 100.0, 20.0, 20.0, 1.0, 1.0])


def _observe_intersection(
    state: IntersectionEnvState,
    params: IntersectionEnvParams,
) -> jnp.ndarray:
    """Compute absolute-coordinate observation for intersection.

    Returns:
        Flat vector of shape (N_OBS_INTERSECTION * N_FEATURES_INTERSECTION,) = (105,).
    """
    vehicles = state.vehicles

    ego_x = vehicles.x[0]
    ego_y = vehicles.y[0]

    # Features: [presence, x, y, vx, vy, cos_h, sin_h]
    # NOTE: vehicles.vx stores scalar speed magnitude; the world-frame
    # velocity components for the observation are derived from heading.
    presence = vehicles.active.astype(jnp.float32)
    cos_h = jnp.cos(vehicles.heading)
    sin_h = jnp.sin(vehicles.heading)
    speed = vehicles.vx  # bicycle-model speed magnitude
    world_vx = speed * cos_h
    world_vy = speed * sin_h

    features = jnp.stack([
        presence,
        vehicles.x,
        vehicles.y,
        world_vx,
        world_vy,
        cos_h,
        sin_h,
    ], axis=-1)  # (N_MAX, 7)

    # Sort by distance to ego
    dist_sq = (vehicles.x - ego_x) ** 2 + (vehicles.y - ego_y) ** 2
    dist_sq = dist_sq.at[0].set(-1.0)  # ego first
    dist_sq = jnp.where(vehicles.active, dist_sq, 1e10)

    sorted_indices = jnp.argsort(dist_sq)
    indices = sorted_indices[:N_OBS_INTERSECTION]

    obs_features = features[indices]  # (15, 7)

    # Normalize
    obs_normalized = obs_features / _INTERSECTION_SCALES[None, :]

    return obs_normalized.flatten()  # (105,)


# ======================================================================
#  Reward
# ======================================================================

def _compute_intersection_reward(
    state: IntersectionEnvState,
    action: jnp.ndarray,
    params: IntersectionEnvParams,
) -> jnp.ndarray:
    """Compute intersection reward."""
    crashed = state.crashed.astype(jnp.float32)
    arrived = state.arrived.astype(jnp.float32)

    # Speed component
    speed = jnp.sqrt(state.vehicles.vx[0] ** 2 + state.vehicles.vy[0] ** 2)
    scaled_speed = lmap(
        speed,
        (params.reward_speed_range_low, params.reward_speed_range_high),
        (0.0, 1.0),
    )
    scaled_speed = jnp.clip(scaled_speed, 0.0, 1.0)

    reward = (
        params.collision_reward * crashed
        + params.high_speed_reward * scaled_speed
        + params.arrived_reward * arrived
    )

    # Gym override: when arrived, reward = arrived_reward (flat, not additive)
    reward = jnp.where(arrived > 0.5, params.arrived_reward, reward)

    # Normalize
    reward = jnp.where(
        params.normalize_reward,
        lmap(
            reward,
            (params.collision_reward,
             params.arrived_reward),
            (0.0, 1.0),
        ),
        reward,
    )

    # On-road gate (match Gym behavior)
    # Intersection road is cross-shaped; approximate with generous bounds.
    ego_x = state.vehicles.x[0]
    ego_y = state.vehicles.y[0]
    bound = params.road_length + params.intersection_size
    in_bounds = (jnp.abs(ego_x) < bound) & (jnp.abs(ego_y) < bound)
    # Check roughly on a road segment (within lane_width of an axis or in intersection)
    in_intersection = (jnp.abs(ego_x) < params.intersection_size) & (jnp.abs(ego_y) < params.intersection_size)
    on_ns_road = (jnp.abs(ego_x) < params.lane_width * 2) & in_bounds
    on_ew_road = (jnp.abs(ego_y) < params.lane_width * 2) & in_bounds
    is_on_road = (in_intersection | on_ns_road | on_ew_road).astype(jnp.float32)
    reward = reward * is_on_road

    return reward
