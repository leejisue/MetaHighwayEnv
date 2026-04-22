"""RoundaboutJaxEnv — pure-functional JAX roundabout environment.

Models the roundabout as a wrapped road (vehicles circulate):
  - Road is a straight segment of length ``road_length ≈ 2*pi*radius ≈ 126 m``
  - X coordinates wrap with modular arithmetic at ``road_length``
  - Observation uses EGO-RELATIVE wrapped x coordinates (continuous)

Follows the gymnax pattern:
  - ``reset(key, params) -> (obs, state)``
  - ``step(key, state, action, params) -> (obs, state, reward, done, info)``
  - ``step_auto_reset(...)`` — auto-reset on done

All methods are static and JIT-compatible.
"""

from __future__ import annotations

from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import jax.random as jrandom

from .state import VehicleState, EnvState, RoundaboutEnvParams, N_MAX, N_OBS, N_SIM_PER_POLICY
from .kinematics import update_vehicles_kinematics
from .collision import check_ego_collision
from .traffic import traffic_step
from .lane import lane_center_y, get_lane_id, on_road
from .utils import lmap
from .discrete_action import (
    apply_meta_action,
    compute_ego_control,
    ROUNDABOUT_TARGET_SPEEDS,
    N_DISCRETE_ACTIONS,
)


class RoundaboutJaxEnv:
    """Pure-functional JAX roundabout environment.

    The roundabout is modelled as a wrapped straight road of length
    ``road_length ≈ 2*pi*radius``.  When a vehicle reaches ``road_length``
    its x coordinate wraps to 0, producing circular circulation naturally.

    All methods are static — no mutable self state.
    Compatible with ``jax.jit``, ``jax.vmap``, ``jax.lax.scan``.
    """

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def reset(
        key: jnp.ndarray,
        params: RoundaboutEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState]:
        """Initialise a fresh episode.

        1. Place ego at x=0, lane 0, with ``ego_initial_speed``.
        2. Spread ``params.num_vehicles - 1`` traffic vehicles randomly
           around the wrapped road across all lanes.

        Returns:
            (obs, state)
        """
        key_traffic, key_speed, key_lanes = jrandom.split(key, 3)

        n_vehicles = params.num_vehicles  # total including ego
        n_active = jnp.minimum(n_vehicles, N_MAX)

        # --- Ego vehicle (index 0) ---
        ego_lane = 0
        ego_y = lane_center_y(jnp.array(ego_lane), params)
        ego_x = 0.0

        # --- Traffic vehicles (indices 1..n_active-1) ---
        traffic_lanes = jrandom.randint(
            key_lanes, (N_MAX - 1,), 0, params.num_lanes
        )
        traffic_y = lane_center_y(traffic_lanes, params)

        # Spread vehicles uniformly around the wrapped road
        spacing = params.road_length / jnp.maximum(n_active - 1, 1)
        base_x = jnp.arange(1, N_MAX) * spacing
        x_noise = jrandom.uniform(
            key_traffic, (N_MAX - 1,),
            minval=-spacing * 0.3, maxval=spacing * 0.3,
        )
        traffic_x = (base_x + x_noise) % params.road_length

        # Random speeds around idm_desired_speed
        speed_noise = jrandom.uniform(
            key_speed, (N_MAX - 1,), minval=-3.0, maxval=3.0
        )
        traffic_speeds = jnp.clip(
            params.idm_desired_speed + speed_noise, 2.0, params.max_speed
        )

        # --- Assemble full arrays ---
        x = jnp.concatenate([jnp.array([ego_x]), traffic_x])
        y = jnp.concatenate([jnp.array([ego_y]), traffic_y])
        vx = jnp.concatenate([jnp.array([params.ego_initial_speed]), traffic_speeds])
        vy = jnp.zeros(N_MAX)
        heading = jnp.zeros(N_MAX)

        all_lanes = jnp.concatenate([
            jnp.array([ego_lane], dtype=jnp.int32),
            traffic_lanes,
        ])

        target_speeds = jnp.concatenate([
            jnp.array([params.ego_initial_speed]),
            traffic_speeds,
        ])

        # Active mask: first n_active are True
        active = jnp.arange(N_MAX) < n_active

        length = jnp.full(N_MAX, params.vehicle_length)
        width = jnp.full(N_MAX, params.vehicle_width)

        vehicles = VehicleState(
            x=x, y=y, vx=vx, vy=vy,
            heading=heading,
            active=active,
            lane_id=all_lanes,
            target_lane_id=all_lanes,
            length=length,
            width=width,
            target_speed=target_speeds,
        )

        state = EnvState(
            vehicles=vehicles,
            time=jnp.array(0.0),
            step_count=jnp.array(0, dtype=jnp.int32),
            crashed=jnp.array(False),
            truncated=jnp.array(False),
            done=jnp.array(False),
        )

        obs = RoundaboutJaxEnv.observe_absolute(state, params)
        return obs, state

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def step(
        key: jnp.ndarray,
        state: EnvState,
        action: jnp.ndarray,
        params: RoundaboutEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Execute one policy step (may include multiple simulation sub-steps).

        Args:
            key: PRNG key.
            state: current environment state.
            action: (2,) array [acceleration, steering].
            params: environment parameters.

        Returns:
            (obs, next_state, reward, done, info)
        """
        ego_accel = action[0]
        ego_steering = action[1]

        def _sim_step(carry, _):
            """Single simulation sub-step with x-wrapping for circular road."""
            veh, crashed, sim_key = carry
            sim_key, traffic_key = jrandom.split(sim_key)

            # 1. Traffic: IDM + MOBIL for non-ego vehicles
            veh, traffic_accels, traffic_steerings = traffic_step(veh, params, traffic_key)

            # 2. Combine: ego action at index 0, traffic for rest
            accels = traffic_accels.at[0].set(ego_accel)
            steerings = traffic_steerings.at[0].set(ego_steering)

            # 3. Kinematics update (all vehicles)
            veh = update_vehicles_kinematics(
                veh, accels, steerings, params.dt, params.max_speed,
            )

            # 4. Wrap x coordinates: circular road
            new_x = veh.x % params.road_length
            veh = veh.replace(x=new_x)

            # 5. Update lane IDs from y position
            new_lane_ids = get_lane_id(veh.y, params)
            veh = veh.replace(lane_id=new_lane_ids)

            # 6. Check if lane change is complete (snap target if close)
            at_target = jnp.abs(
                veh.y - lane_center_y(veh.target_lane_id, params)
            ) < params.lane_width * 0.3
            new_targets = jnp.where(at_target, new_lane_ids, veh.target_lane_id)
            veh = veh.replace(target_lane_id=new_targets)

            # 7. Collision check
            ego_crashed = check_ego_collision(veh)
            crashed = crashed | ego_crashed

            return (veh, crashed, sim_key), None

        (vehicles, crashed, _), _ = jax.lax.scan(
            _sim_step,
            (state.vehicles, state.crashed, key),
            None,
            length=N_SIM_PER_POLICY,
        )

        new_time = state.time + 1.0
        new_step = state.step_count + 1
        truncated = new_step >= params.max_steps
        done = crashed | truncated

        new_state = EnvState(
            vehicles=vehicles,
            time=new_time,
            step_count=new_step,
            crashed=crashed,
            truncated=truncated,
            done=done,
        )

        obs = RoundaboutJaxEnv.observe_absolute(new_state, params)
        reward = RoundaboutJaxEnv._compute_reward(new_state, action, params)

        info = {
            "crashed": crashed,
            "truncated": truncated,
            "step_count": new_step,
        }

        return obs, new_state, reward, done, info

    # ------------------------------------------------------------------
    #  Auto-reset step (gymnax pattern)
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def step_auto_reset(
        key: jnp.ndarray,
        state: EnvState,
        action: jnp.ndarray,
        params: RoundaboutEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Step with automatic reset on done (gymnax pattern).

        When the episode ends (done=True), the returned obs and state
        are from a fresh ``reset()`` call, but the reward and done
        correspond to the terminal transition.

        Returns:
            (obs, next_state, reward, done, info)
        """
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = RoundaboutJaxEnv.step(
            key_step, state, action, params,
        )

        # FIX P1-2 FINAL: use lax.cond so reset() only executes on done=True.
        # The old pattern always called reset() unconditionally. lax.cond only
        # traces and executes one branch at runtime, avoiding wasted work ~97.5% of steps.
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: RoundaboutJaxEnv.reset(key_reset, params),
            false_fun=lambda _: (obs, next_state),
            operand=None,
        )

        # If done, swap in the reset state/obs
        next_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(done, a, b),
            state_reset,
            next_state,
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
        state: EnvState,
        action: jnp.ndarray,
        params: RoundaboutEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Execute one policy step with a discrete meta-action.

        Matches Highway-Env's ``DiscreteMetaAction`` with 5 actions:
          0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER.

        Target speeds: [0, 8, 16] m/s (from roundabout_env.py).

        Args:
            key: PRNG key.
            state: current environment state.
            action: scalar integer in {0, 1, 2, 3, 4}.
            params: environment parameters.

        Returns:
            (obs, next_state, reward, done, info)
        """
        # 1. Apply meta-action
        vehicles = apply_meta_action(
            action, state.vehicles, params.num_lanes, ROUNDABOUT_TARGET_SPEEDS,
        )
        state = state.replace(vehicles=vehicles)

        # 2. Simulation loop with x-wrapping — recompute PD control every
        #    sub-step so heading tracks the lane center smoothly (holding
        #    constant control across all 15 sub-steps caused visible wobble).
        def _sim_step(carry, _):
            veh, crashed, sim_key = carry
            sim_key, traffic_key = jrandom.split(sim_key)

            veh, traffic_accels, traffic_steerings = traffic_step(veh, params, traffic_key)

            ego_accel, ego_steering = compute_ego_control(veh, params)
            accels = traffic_accels.at[0].set(ego_accel)
            steerings = traffic_steerings.at[0].set(ego_steering)

            veh = update_vehicles_kinematics(
                veh, accels, steerings, params.dt, params.max_speed,
            )

            # Wrap x for circular road
            new_x = veh.x % params.road_length
            veh = veh.replace(x=new_x)

            new_lane_ids = get_lane_id(veh.y, params)
            veh = veh.replace(lane_id=new_lane_ids)

            at_target = jnp.abs(
                veh.y - lane_center_y(veh.target_lane_id, params)
            ) < params.lane_width * 0.3
            new_targets = jnp.where(at_target, new_lane_ids, veh.target_lane_id)
            veh = veh.replace(target_lane_id=new_targets)

            ego_crashed = check_ego_collision(veh)
            crashed = crashed | ego_crashed

            return (veh, crashed, sim_key), None

        (vehicles, crashed, _), _ = jax.lax.scan(
            _sim_step,
            (state.vehicles, state.crashed, key),
            None,
            length=N_SIM_PER_POLICY,
        )

        new_time = state.time + 1.0
        new_step = state.step_count + 1
        truncated = new_step >= params.max_steps
        done = crashed | truncated

        new_state = EnvState(
            vehicles=vehicles,
            time=new_time,
            step_count=new_step,
            crashed=crashed,
            truncated=truncated,
            done=done,
        )

        obs = RoundaboutJaxEnv.observe_absolute(new_state, params)
        ego_accel_final, ego_steering_final = compute_ego_control(vehicles, params)
        ctrl_action = jnp.array([ego_accel_final, ego_steering_final])
        # Action-based lane-change detection (matching Gym and highway fix)
        is_lane_change = ((action == 0) | (action == 2)).astype(jnp.float32)
        reward = RoundaboutJaxEnv._compute_reward(new_state, ctrl_action, params,
                                                   is_lane_change=is_lane_change)

        info = {
            "crashed": crashed,
            "truncated": truncated,
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
        state: EnvState,
        action: jnp.ndarray,
        params: RoundaboutEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Discrete meta-action step with automatic reset on done."""
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = RoundaboutJaxEnv.step_discrete(
            key_step, state, action, params,
        )

        # FIX P1-2 FINAL: lazy reset via lax.cond (same as step_auto_reset)
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: RoundaboutJaxEnv.reset(key_reset, params),
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
    #  Observation: absolute coordinates
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def observe_absolute(
        state: EnvState,
        params: RoundaboutEnvParams,
    ) -> jnp.ndarray:
        """Compute kinematic observation with EGO-RELATIVE x coordinates.

        Features per vehicle: [presence, dx_rel, y, vx, vy].
        The x-coordinate uses ego-relative wrapped distance to eliminate
        the discontinuity at the road_length wrap boundary.  Without this,
        a vehicle at x=125 (near road_length=126) and one at x=1 appear
        far apart in absolute coords but are actually adjacent.

        Normalisation:
          - dx_rel  divided by ``road_length / 2`` (ego = 0)
          - y       divided by 100
          - vx      divided by ``max_speed``
          - vy      divided by ``max_speed``

        Vehicles (excluding ego) are sorted by wrapped circular distance to
        ego.  The ego is always placed in row 0.

        Returns:
            Flat array of shape ``(N_OBS * 5,)`` = 25 dimensions.
        """
        vehicles = state.vehicles

        ego_x = vehicles.x[0]
        ego_y = vehicles.y[0]
        ego_vx = vehicles.vx[0]
        ego_vy = vehicles.vy[0]

        # ------ Sorting by wrapped distance (skip ego at index 0) ------
        dx = vehicles.x - ego_x
        # Shortest path around the circle
        dx_wrapped = jnp.where(
            jnp.abs(dx) > params.road_length / 2,
            dx - jnp.sign(dx) * params.road_length,
            dx,
        )
        dy = vehicles.y - ego_y
        dist_sq = dx_wrapped ** 2 + dy ** 2

        # Ego always first; inactive vehicles pushed to the back
        dist_sq = dist_sq.at[0].set(-1.0)
        dist_sq = jnp.where(vehicles.active, dist_sq, 1e10)

        sorted_indices = jnp.argsort(dist_sq)
        indices = sorted_indices[:N_OBS]  # (N_OBS,)

        # ------ Build feature matrix ------
        presence = vehicles.active.astype(jnp.float32)  # (N_MAX,)

        # Normalisation denominators
        x_scale = params.road_length / 2.0
        y_scale = 100.0
        v_scale = params.max_speed

        # Ego-relative x: use wrapped distance (continuous, no discontinuity)
        x_norm = dx_wrapped / x_scale  # ego=0, others in [-1, +1]
        y_norm = vehicles.y / y_scale
        vx_norm = vehicles.vx / v_scale
        vy_norm = vehicles.vy / v_scale

        features = jnp.stack(
            [presence, x_norm, y_norm, vx_norm, vy_norm], axis=-1
        )  # (N_MAX, 5)

        # Gather top-N_OBS vehicles
        obs_features = features[indices]  # (N_OBS, 5)

        # Ego row: presence=1, dx_rel=0 (relative to self), absolute y/v
        ego_row = jnp.array([
            1.0,
            0.0,  # ego-relative x = 0
            ego_y / y_scale,
            ego_vx / v_scale,
            ego_vy / v_scale,
        ])
        obs_features = obs_features.at[0].set(ego_row)

        return obs_features.flatten()  # (N_OBS * 5,)

    # ------------------------------------------------------------------
    #  Reward (internal helper)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reward(
        state: EnvState,
        action: jnp.ndarray,
        params: RoundaboutEnvParams,
        is_lane_change: jnp.ndarray = 0.0,
    ) -> jnp.ndarray:
        """Compute roundabout reward (scalar).

        Components match the Gym roundabout reward structure:
          - collision_reward  : -1 if crashed
          - high_speed_reward : scaled speed [0, 1] (Gym uses [0, MAX_SPEED])
          - right_lane_reward : lane fraction [0, 1]
          - lane_change_reward: action-based detection (matching Gym)
        """
        vehicles = state.vehicles

        # Collision
        crashed = state.crashed.astype(jnp.float32)

        # Speed — match Gym: lmap(vehicle.speed, [0, MAX_SPEED], [0, 1])
        ego_speed = jnp.sqrt(vehicles.vx[0] ** 2 + vehicles.vy[0] ** 2)
        scaled_speed = lmap(
            ego_speed,
            (0.0, params.max_speed),
            (0.0, 1.0),
        )
        scaled_speed = jnp.clip(scaled_speed, 0.0, 1.0)

        # Right (inner) lane
        lane_id = vehicles.lane_id[0].astype(jnp.float32)
        max_lane = jnp.maximum(params.num_lanes - 1, 1).astype(jnp.float32)
        right_lane_frac = lane_id / max_lane

        # Lane-change penalty: action-based (matching Gym)
        lane_change = is_lane_change

        # On-road gate
        is_on_road = on_road(vehicles.y[0], params).astype(jnp.float32)

        reward = (
            params.collision_reward * crashed
            + params.high_speed_reward * scaled_speed
            + params.right_lane_reward * right_lane_frac
            + params.lane_change_reward * lane_change
        )

        # Normalise — match Gym: [collision_reward, high_speed_reward] -> [0, 1]
        reward = jnp.where(
            params.normalize_reward,
            lmap(
                reward,
                (params.collision_reward,
                 params.high_speed_reward),
                (0.0, 1.0),
            ),
            reward,
        )

        reward = reward * is_on_road
        return reward
