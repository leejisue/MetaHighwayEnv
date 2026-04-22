"""HighwayJaxEnv — pure-functional JAX highway environment.

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

from .state import VehicleState, EnvState, EnvParams, N_MAX, N_SIM_PER_POLICY
from .kinematics import update_ego, update_vehicles_kinematics
from .lane import lane_center_y, get_lane_id, on_road
from .traffic import traffic_step
from .collision import check_ego_collision
from .observation import observe
from .reward import compute_reward
from .discrete_action import (
    apply_meta_action,
    compute_ego_control,
    HIGHWAY_TARGET_SPEEDS,
    N_DISCRETE_ACTIONS,
)


class HighwayJaxEnv:
    """Pure-functional JAX highway environment.

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
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, EnvState]:
        """Initialise a fresh episode.

        1. Place ego at lane 0, x=0, with initial speed.
        2. Spawn ``params.num_vehicles`` traffic vehicles at random
           positions across all lanes.

        Returns:
            (obs, state)
        """
        key_ego, key_traffic, key_speed, key_lanes = jrandom.split(key, 4)

        n_vehicles = params.num_vehicles  # total including ego
        # Clamp to N_MAX
        n_active = jnp.minimum(n_vehicles, N_MAX)

        # --- Ego vehicle (index 0) ---
        ego_lane = 0
        ego_y = lane_center_y(jnp.array(ego_lane), params)
        ego_x = 0.0

        # --- Traffic vehicles (indices 1..n_active-1) ---
        # Random lane assignment
        traffic_lanes = jrandom.randint(
            key_lanes, (N_MAX - 1,), 0, params.num_lanes
        )
        traffic_y = lane_center_y(traffic_lanes, params)

        # Spread vehicles along the road with spacing based on density
        spacing = params.vehicle_length * 2.0 / jnp.maximum(params.vehicles_density, 0.1)
        # Random offsets around regular spacing
        base_x = jnp.arange(1, N_MAX) * spacing
        x_noise = jrandom.uniform(key_traffic, (N_MAX - 1,), minval=-spacing * 0.3, maxval=spacing * 0.3)
        traffic_x = base_x + x_noise

        # Random target speeds around desired speed
        speed_noise = jrandom.uniform(
            key_speed, (N_MAX - 1,), minval=-5.0, maxval=5.0
        )
        traffic_speeds = jnp.clip(
            params.idm_desired_speed + speed_noise, 15.0, params.max_speed
        )

        # FIX P1-3: replace jnp.concatenate([jnp.array([scalar]), vec]) with
        # pre-allocated (N_MAX,) arrays written via .at[0].set() / .at[1:].set().
        # The old pattern created 4 temporary 1-element arrays + 4 concat ops
        # each reset call. reset() runs every step_auto_reset call (unconditional),
        # so this saves 8 device allocations per policy step.
        x = jnp.zeros(N_MAX).at[0].set(ego_x).at[1:].set(traffic_x)
        y = jnp.zeros(N_MAX).at[0].set(ego_y).at[1:].set(traffic_y)
        vx = jnp.zeros(N_MAX).at[0].set(params.ego_initial_speed).at[1:].set(traffic_speeds)
        vy = jnp.zeros(N_MAX)
        heading = jnp.zeros(N_MAX)

        all_lanes = jnp.zeros(N_MAX, dtype=jnp.int32).at[0].set(
            jnp.array(ego_lane, dtype=jnp.int32)
        ).at[1:].set(traffic_lanes)

        target_speeds = jnp.zeros(N_MAX).at[0].set(
            params.ego_initial_speed
        ).at[1:].set(traffic_speeds)

        # Active mask: first n_active are True
        active = jnp.arange(N_MAX) < n_active

        length = jnp.full(N_MAX, params.vehicle_length)
        width = jnp.full(N_MAX, params.vehicle_width)

        vehicles = VehicleState(
            x=x, y=y, vx=vx, vy=vy,
            heading=heading,
            active=active,
            lane_id=all_lanes,
            target_lane_id=all_lanes,  # initially target = current
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

        obs = observe(state, params)
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
        params: EnvParams,
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
            """Single simulation sub-step."""
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

            # 4. Update lane IDs from y position
            new_lane_ids = get_lane_id(veh.y, params)
            veh = veh.replace(lane_id=new_lane_ids)

            # 5. Check if lane change is complete (snap target if close)
            at_target = jnp.abs(
                veh.y - lane_center_y(veh.target_lane_id, params)
            ) < params.lane_width * 0.3
            new_targets = jnp.where(at_target, new_lane_ids, veh.target_lane_id)
            veh = veh.replace(target_lane_id=new_targets)

            # 6. Collision check
            ego_crashed = check_ego_collision(veh)
            crashed = crashed | ego_crashed

            return (veh, crashed, sim_key), None

        (vehicles, crashed, _), _ = jax.lax.scan(
            _sim_step,
            (state.vehicles, state.crashed, key),
            None,
            length=N_SIM_PER_POLICY,
        )

        # Update state (policy_frequency=1 → 1 second per policy step)
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

        obs = observe(new_state, params)
        reward = compute_reward(new_state, action, params)

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
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Step with automatic reset on done (gymnax pattern).

        When the episode ends (done=True), the returned obs and state
        are from a fresh ``reset()`` call, but the reward and done
        correspond to the terminal transition.

        Returns:
            (obs, next_state, reward, done, info)
        """
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = HighwayJaxEnv.step(
            key_step, state, action, params,
        )

        # FIX P1-2: use lax.cond so reset() only executes on done=True.
        # The old pattern always called reset() (both branches of jnp.where
        # execute in JAX). reset() includes random sampling + array allocation
        # that is wasted ~97.5% of steps (episode length=40, done rate=1/40).
        # lax.cond only traces and executes one branch at runtime.
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: HighwayJaxEnv.reset(key_reset, params),
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
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Execute one policy step with a discrete meta-action.

        Matches Highway-Env's ``DiscreteMetaAction`` with 5 actions:
          0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER.

        The action updates the ego's target lane and target speed,
        then a PD controller computes continuous acceleration/steering
        for the kinematic bicycle model.

        Args:
            key: PRNG key.
            state: current environment state.
            action: scalar integer in {0, 1, 2, 3, 4}.
            params: environment parameters.

        Returns:
            (obs, next_state, reward, done, info)
        """
        # 1. Apply meta-action to update ego targets
        vehicles = apply_meta_action(
            action, state.vehicles, params.num_lanes, HIGHWAY_TARGET_SPEEDS,
        )
        state = state.replace(vehicles=vehicles)

        # 2. Run the simulation loop (PD controller is recomputed every
        #    sub-step using the latest ego state — holding control commands
        #    constant across 15 sub-steps caused visible heading wobble.)
        def _sim_step(carry, _):
            veh, crashed, sim_key = carry
            sim_key, traffic_key = jrandom.split(sim_key)

            veh, traffic_accels, traffic_steerings = traffic_step(veh, params, traffic_key)

            # Recompute ego control from current heading/lateral error
            ego_accel, ego_steering = compute_ego_control(veh, params)
            accels = traffic_accels.at[0].set(ego_accel)
            steerings = traffic_steerings.at[0].set(ego_steering)

            veh = update_vehicles_kinematics(
                veh, accels, steerings, params.dt, params.max_speed,
            )

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

        obs = observe(new_state, params)
        # For reward, recompute the controller output from the final state
        # so the action passed to the reward function reflects the last
        # commanded acceleration/steering.
        ego_accel_final, ego_steering_final = compute_ego_control(vehicles, params)
        ctrl_action = jnp.array([ego_accel_final, ego_steering_final])
        # Detect lane-change action: 0=LANE_LEFT, 2=LANE_RIGHT
        is_lane_change = ((action == 0) | (action == 2)).astype(jnp.float32)
        reward = compute_reward(new_state, ctrl_action, params,
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
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Discrete meta-action step with automatic reset on done."""
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = HighwayJaxEnv.step_discrete(
            key_step, state, action, params,
        )

        # FIX P1-2: lazy reset via lax.cond (same as step_auto_reset)
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: HighwayJaxEnv.reset(key_reset, params),
            false_fun=lambda _: (obs, next_state),
            operand=None,
        )

        next_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(done, a, b),
            state_reset,
            next_state,
        )
        obs = jnp.where(done, obs_reset, obs)

        return obs, next_state, reward, done, info
