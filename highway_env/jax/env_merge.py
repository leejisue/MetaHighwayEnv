"""MergeJaxEnv — pure-functional JAX merge environment.

Follows the gymnax pattern:
  - ``reset(key, params) -> (obs, state)``
  - ``step(key, state, action, params) -> (obs, state, reward, done, info)``
  - ``step_auto_reset(...)`` — auto-reset on done

The road has ``params.num_lanes`` main lanes plus one merge lane
(lane_id == num_lanes).  Merge vehicles spawn in the merge lane
between ``merge_start_x`` and ``merge_end_x`` and use MOBIL to
naturally try to change into lane ``num_lanes - 1``.

All methods are static and JIT-compatible.
"""

from __future__ import annotations

from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import jax.random as jrandom

from .state import VehicleState, EnvState, MergeEnvParams, N_MAX, N_OBS, N_SIM_PER_POLICY
from .kinematics import update_vehicles_kinematics
from .lane import lane_center_y, get_lane_id, on_road
from .traffic import traffic_step
from .collision import check_ego_collision
from .observation import observe
from .utils import lmap
from .discrete_action import (
    apply_meta_action,
    compute_ego_control,
    MERGE_TARGET_SPEEDS,
    N_DISCRETE_ACTIONS,
)


class MergeJaxEnv:
    """Pure-functional JAX merge environment.

    All methods are static — no mutable self state.
    Compatible with ``jax.jit``, ``jax.vmap``, ``jax.lax.scan``.

    The environment adds a merge lane (lane_id == num_lanes) on top of
    the standard highway.  Merge vehicles spawn between ``merge_start_x``
    and ``merge_end_x`` and are incentivised by MOBIL to change into the
    rightmost main lane.  A ``merging_speed_reward`` penalty is applied
    when merge-zone vehicles move slowly.
    """

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    @staticmethod
    @jax.jit
    def reset(
        key: jnp.ndarray,
        params: MergeEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState]:
        """Initialise a fresh episode.

        1. Place ego at lane 1, x=0, with initial speed.
        2. Spawn ``params.num_vehicles - params.num_merge_vehicles`` highway
           traffic vehicles at random positions across the main lanes.
        3. Spawn ``params.num_merge_vehicles`` vehicles in the merge lane
           (lane_id == num_lanes) at x positions inside the merge zone.

        Returns:
            (obs, state)
        """
        key_ego, key_hw, key_merge, key_speed_hw, key_speed_mg, key_lanes = jrandom.split(key, 6)

        n_hw = params.num_vehicles - params.num_merge_vehicles  # highway traffic count
        n_mg = params.num_merge_vehicles                         # merge lane count
        n_total = params.num_vehicles                            # ego not counted in these

        # Total active slots: 1 (ego) + n_total traffic
        n_active = jnp.minimum(n_total + 1, N_MAX)

        # --- Ego vehicle (index 0): lane 1, x=0 ---
        ego_lane = 1
        ego_y = lane_center_y(jnp.array(ego_lane, dtype=jnp.int32), params)
        ego_x = 0.0

        # --- Highway traffic (indices 1..n_hw): random main lanes ---
        hw_lanes = jrandom.randint(key_lanes, (N_MAX - 1,), 0, params.num_lanes)
        hw_y = lane_center_y(hw_lanes, params)

        spacing = params.vehicle_length * 2.0 / jnp.maximum(params.vehicles_density, 0.1)
        base_x = jnp.arange(1, N_MAX) * spacing
        x_noise_hw = jrandom.uniform(key_hw, (N_MAX - 1,), minval=-spacing * 0.3, maxval=spacing * 0.3)
        hw_x = base_x + x_noise_hw

        speed_noise_hw = jrandom.uniform(key_speed_hw, (N_MAX - 1,), minval=-5.0, maxval=5.0)
        hw_speeds = jnp.clip(params.idm_desired_speed + speed_noise_hw, 15.0, params.max_speed)

        # --- Merge lane vehicles: placed in [merge_start_x, merge_end_x] ---
        # We reuse the last n_mg slots of the traffic arrays.
        merge_zone_len = jnp.maximum(params.merge_end_x - params.merge_start_x, 1.0)
        merge_offsets = jrandom.uniform(key_merge, (N_MAX - 1,), minval=0.0, maxval=1.0)
        merge_xs = params.merge_start_x + merge_offsets * merge_zone_len

        speed_noise_mg = jrandom.uniform(key_speed_mg, (N_MAX - 1,), minval=-3.0, maxval=3.0)
        merge_speeds = jnp.clip(params.idm_desired_speed + speed_noise_mg, 10.0, params.max_speed)

        merge_lane_id = params.num_lanes  # merge lane index
        merge_y_val = lane_center_y(jnp.array(merge_lane_id, dtype=jnp.int32), params)
        merge_ys = jnp.full(N_MAX - 1, merge_y_val)

        # --- Combine traffic arrays ---
        # Indices 1..n_hw are highway, indices n_hw+1..n_hw+n_mg are merge.
        # We build index arrays and use jnp.where to select per slot.
        slot_idx = jnp.arange(N_MAX - 1)  # 0-based within traffic slots
        is_merge_slot = slot_idx >= n_hw

        combined_x = jnp.where(is_merge_slot, merge_xs, hw_x)
        combined_y = jnp.where(is_merge_slot, merge_ys, hw_y)
        combined_vx = jnp.where(is_merge_slot, merge_speeds, hw_speeds)
        combined_lanes = jnp.where(
            is_merge_slot,
            jnp.full(N_MAX - 1, merge_lane_id, dtype=jnp.int32),
            hw_lanes,
        )

        # --- Assemble full arrays (index 0 = ego) ---
        x = jnp.concatenate([jnp.array([ego_x]), combined_x])
        y = jnp.concatenate([jnp.array([ego_y]), combined_y])
        vx = jnp.concatenate([jnp.array([params.ego_initial_speed]), combined_vx])
        vy = jnp.zeros(N_MAX)
        heading = jnp.zeros(N_MAX)
        all_lanes = jnp.concatenate([
            jnp.array([ego_lane], dtype=jnp.int32),
            combined_lanes,
        ])

        target_speeds = jnp.concatenate([
            jnp.array([params.ego_initial_speed]),
            combined_vx,
        ])

        # Active mask: first n_active slots are True
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
        params: MergeEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Execute one policy step (may include multiple simulation sub-steps).

        Args:
            key: PRNG key.
            state: current environment state.
            action: (2,) array [acceleration, steering].
            params: merge environment parameters.

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
            #    Merge-lane vehicles (lane_id == num_lanes) will attempt to
            #    change into lane num_lanes-1 via MOBIL naturally.
            veh, traffic_accels, traffic_steerings = traffic_step(veh, params, traffic_key)

            # 2. Override ego with policy action
            accels = traffic_accels.at[0].set(ego_accel)
            steerings = traffic_steerings.at[0].set(ego_steering)

            # 3. Kinematics update (all vehicles)
            veh = update_vehicles_kinematics(
                veh, accels, steerings, params.dt, params.max_speed,
            )

            # 4. Update lane IDs from y position
            #    Merge env has lane_id = num_lanes for the merge lane,
            #    so clip to [0, num_lanes] (inclusive) instead of [0, num_lanes-1].
            raw_lane = jnp.floor(veh.y / params.lane_width).astype(jnp.int32)
            new_lane_ids = jnp.clip(raw_lane, 0, params.num_lanes)
            veh = veh.replace(lane_id=new_lane_ids)

            # 5. Snap target lane if lane change is complete
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

        # Update time / step counters
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
        reward = _compute_merge_reward(new_state, action, params)

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
        params: MergeEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Step with automatic reset on done (gymnax pattern).

        When the episode ends (done=True), the returned obs and state
        are from a fresh ``reset()`` call, but the reward and done
        correspond to the terminal transition.

        Returns:
            (obs, next_state, reward, done, info)
        """
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = MergeJaxEnv.step(
            key_step, state, action, params,
        )

        # FIX P1-2 FINAL: use lax.cond so reset() only executes on done=True.
        # The old pattern always called reset() unconditionally (both branches of
        # jnp.where execute in JAX). lax.cond only traces and executes one branch
        # at runtime, avoiding wasted random sampling + array allocation ~97.5% of steps.
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: MergeJaxEnv.reset(key_reset, params),
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
        params: MergeEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Execute one policy step with a discrete meta-action.

        Matches Highway-Env's ``DiscreteMetaAction`` with 5 actions:
          0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER.

        Args:
            key: PRNG key.
            state: current environment state.
            action: scalar integer in {0, 1, 2, 3, 4}.
            params: merge environment parameters.

        Returns:
            (obs, next_state, reward, done, info)
        """
        # 1. Apply meta-action (lane clamped to main lanes only, not merge lane)
        vehicles = apply_meta_action(
            action, state.vehicles, params.num_lanes, MERGE_TARGET_SPEEDS,
        )
        state = state.replace(vehicles=vehicles)

        # 2. Simulation loop — recompute PD control every sub-step so the
        #    ego heading stays smooth (holding control constant across all
        #    15 sub-steps caused a visible wobble in rendered videos).
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

            raw_lane = jnp.floor(veh.y / params.lane_width).astype(jnp.int32)
            new_lane_ids = jnp.clip(raw_lane, 0, params.num_lanes)
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
        ego_accel_final, ego_steering_final = compute_ego_control(vehicles, params)
        ctrl_action = jnp.array([ego_accel_final, ego_steering_final])
        # Action-based lane-change detection (matching Gym and highway fix)
        is_lane_change = ((action == 0) | (action == 2)).astype(jnp.float32)
        reward = _compute_merge_reward(new_state, ctrl_action, params,
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
        params: MergeEnvParams,
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Discrete meta-action step with automatic reset on done."""
        key_step, key_reset = jrandom.split(key)

        obs, next_state, reward, done, info = MergeJaxEnv.step_discrete(
            key_step, state, action, params,
        )

        # FIX P1-2 FINAL: lazy reset via lax.cond (same as step_auto_reset)
        obs_reset, state_reset = jax.lax.cond(
            done,
            true_fun=lambda _: MergeJaxEnv.reset(key_reset, params),
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
#  Merge reward (module-level helper, not part of class API)
# ------------------------------------------------------------------

def _compute_merge_reward(
    state: EnvState,
    action: jnp.ndarray,
    params: MergeEnvParams,
    is_lane_change: jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Compute the merge reward (scalar).

    Components:
      - collision_reward:     -1 if crashed.
      - high_speed_reward:    ego forward speed scaled to [0, 1].
      - right_lane_reward:    ego lane_id / (num_lanes - 1).
      - lane_change_reward:   penalty when ego changes lane.
      - merging_speed_reward: penalty when merge-zone vehicles are slow.

    When ``normalize_reward`` is True the weighted sum is linearly mapped
    from [worst, best] -> [0, 1] before the on-road gate is applied.
    """
    vehicles = state.vehicles

    # --- Collision ---
    crashed = state.crashed.astype(jnp.float32)

    # --- High-speed (ego) ---
    # FIX S1-1: Gym merge_env.py:101 uses raw `self.vehicle.speed` (scalar magnitude),
    # not forward_speed = speed * cos(heading). Match that formula here.
    forward_speed = vehicles.vx[0]  # raw speed magnitude (vx stores scalar speed in bicycle model)
    scaled_speed = lmap(
        forward_speed,
        (params.reward_speed_range_low, params.reward_speed_range_high),
        (0.0, 1.0),
    )
    scaled_speed = jnp.clip(scaled_speed, 0.0, 1.0)

    # --- Right-lane (ego) ---
    lane_id = vehicles.lane_id[0].astype(jnp.float32)
    max_lane = jnp.maximum(params.num_lanes - 1, 1).astype(jnp.float32)
    right_lane_frac = lane_id / max_lane

    # --- Lane-change penalty (action-based, matching Gym) ---
    lane_change = is_lane_change

    # --- Merging-speed penalty ---
    # FIX S1-2: Gym merge_env.py:115-121 uses:
    #   sum((vehicle.target_speed - vehicle.speed) / vehicle.target_speed
    #       for vehicle in road.vehicles
    #       if vehicle.lane_index == ("b","c",2) and isinstance(vehicle, ControlledVehicle))
    # The gym only looks at ControlledVehicle instances in the merge zone segment.
    # In JAX, all traffic vehicles are ControlledVehicle equivalents; the merge zone
    # corresponds to vehicles in the merge lane (lane_id == num_lanes) within
    # [merge_start_x, merge_end_x]. We sum the relative speed deficit per vehicle
    # to match the gym formula structure (sum, not average).
    in_merge = (
        (vehicles.x > params.merge_start_x)
        & (vehicles.x < params.merge_end_x)
        & (vehicles.lane_id >= params.num_lanes)
        & vehicles.active
    )
    target_speeds = vehicles.target_speed
    # Relative speed deficit: (target - actual) / target, clamped to [0, inf)
    # Use vx as scalar speed magnitude (bicycle model convention)
    speed_deficit = jnp.where(
        in_merge,
        (target_speeds - vehicles.vx) / jnp.maximum(target_speeds, 1e-6),
        0.0,
    )
    # Sum of deficits (matches gym's sum(...) — not averaged)
    merge_speed_deficit_sum = jnp.sum(speed_deficit)
    merge_penalty = params.merging_speed_reward * merge_speed_deficit_sum

    # --- Weighted sum ---
    reward = (
        params.collision_reward * crashed
        + params.high_speed_reward * scaled_speed
        + params.right_lane_reward * right_lane_frac
        + params.lane_change_reward * lane_change
        + merge_penalty
    )

    # --- Normalise ---
    best = params.high_speed_reward + params.right_lane_reward
    worst = params.collision_reward + params.merging_speed_reward
    reward = jnp.where(
        params.normalize_reward,
        lmap(reward, (worst, best), (0.0, 1.0)),
        reward,
    )

    # --- On-road gate ---
    # Merge env has num_lanes main lanes + 1 merge lane, so road extends
    # to (num_lanes + 1) * lane_width instead of num_lanes * lane_width.
    road_top = (params.num_lanes + 1) * params.lane_width
    is_on_road = (
        (vehicles.y[0] >= -params.vehicle_width / 2.0) &
        (vehicles.y[0] <= road_top + params.vehicle_width / 2.0)
    ).astype(jnp.float32)
    reward = reward * is_on_road

    return reward
