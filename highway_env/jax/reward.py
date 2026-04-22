"""Reward function for JAX highway environment.

Ported from ``highway_env.envs.highway_env.HighwayEnv._reward()``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .state import EnvState, EnvParams
from .lane import on_road
from .utils import lmap


def compute_reward(
    state: EnvState,
    action: jnp.ndarray,
    params: EnvParams,
    is_lane_change: jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Compute the highway reward (scalar).

    Reward components (matching original HighwayEnv):
      - collision_reward: -1 if crashed, 0 otherwise (weight: collision_reward)
      - high_speed_reward: scaled forward speed [0, 1] (weight: high_speed_reward)
      - right_lane_reward: lane_id / (num_lanes - 1) [0, 1] (weight: right_lane_reward)
      - lane_change_reward: 1 if lane change action, 0 otherwise (weight: lane_change_reward)
      - on_road_reward: 0 if off-road (multiplicative gate)

    When ``normalize_reward`` is True, the weighted sum is linearly mapped
    to [0, 1].  The normalisation range accounts for negative lane_change_reward.

    Args:
        state: Current environment state (post-step).
        action: Continuous action [acceleration, steering].
        params: Environment parameters.
        is_lane_change: 1.0 if a lane-change action was issued, 0.0 otherwise.
            Passed explicitly by ``step_discrete()`` since the discrete action
            (LANE_LEFT=0, LANE_RIGHT=2) is not available in continuous mode.
            Defaults to 0.0 for backward compatibility.

    Note: ``lane_change_reward`` defaults to 0.0 in EnvParams, so existing
    checkpoints trained without it are unaffected.  Set to a negative value
    (e.g. -0.05) to penalise unnecessary lane changes.
    """
    vehicles = state.vehicles

    # --- Collision component ---
    crashed = state.crashed.astype(jnp.float32)

    # --- Speed component ---
    forward_speed = vehicles.vx[0] * jnp.cos(vehicles.heading[0])
    scaled_speed = lmap(
        forward_speed,
        (params.reward_speed_range_low, params.reward_speed_range_high),
        (0.0, 1.0),
    )
    scaled_speed = jnp.clip(scaled_speed, 0.0, 1.0)

    # --- Right lane component ---
    # FIX S1-5 (CAUTION/TODO): Gym highway_env.py:171-174 uses target_lane_index[2]
    # for ControlledVehicle (anticipates destination lane immediately on action),
    # while JAX uses current lane_id derived from y-position. This causes a reward
    # timing difference during lane-change maneuvers: gym rewards the target lane
    # immediately, JAX rewards it only after the vehicle physically arrives.
    # To fully fix, VehicleState.target_lane_id[0] should be used here.
    # Using target_lane_id when it differs from lane_id (i.e. during a lane change):
    lane_id = vehicles.target_lane_id[0].astype(jnp.float32)
    max_lane = jnp.maximum(params.num_lanes - 1, 1).astype(jnp.float32)
    right_lane_frac = lane_id / max_lane

    # --- On-road gate ---
    is_on_road = on_road(vehicles.y[0], params).astype(jnp.float32)

    # --- Weighted sum ---
    reward = (
        params.collision_reward * crashed
        + params.high_speed_reward * scaled_speed
        + params.right_lane_reward * right_lane_frac
        + params.lane_change_reward * is_lane_change
    )

    # --- Normalise ---
    # Include lane_change_reward in the normalisation range when it is
    # negative (penalty).  When lane_change_reward >= 0, the range stays
    # unchanged for backward compatibility with existing checkpoints.
    norm_high = params.high_speed_reward + params.right_lane_reward
    norm_low = params.collision_reward + jnp.minimum(params.lane_change_reward, 0.0)
    reward = jnp.where(
        params.normalize_reward,
        lmap(
            reward,
            (norm_low, norm_high),
            (0.0, 1.0),
        ),
        reward,
    )

    # Gate by on-road
    reward = reward * is_on_road

    return reward
