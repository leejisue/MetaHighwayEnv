"""MetaHighwayJaxEnv — JAX meta-RL environment for PEARL integration.

Provides task sampling and vectorized rollout collection via
``jax.lax.scan`` and ``jax.vmap``.
"""

from __future__ import annotations

from typing import Tuple, Dict, Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jrandom

# NOTE: vehicle_density range here ([0.5, 2.0]) intentionally differs from
# meta_rl/constants.py DENSITY_RANGES ([0.5, 3.0]) — the JAX sampler uses a
# tighter upper bound to avoid extreme congestion that destabilises JIT-compiled
# rollouts.  Speed and penalty ranges are shared with meta_rl/constants.py.
from highway_env.meta_rl.constants import SPEED_RANGES, PENALTY_RANGES

from .state import EnvState, EnvParams, N_MAX
from .env import HighwayJaxEnv
from .observation import observe


def sample_task_params(
    key: jnp.ndarray,
    base_params: EnvParams,
    vary_vehicle_density: bool = True,
    vary_speed_range: bool = True,
    vary_collision_reward: bool = True,
) -> EnvParams:
    """Sample a random task (EnvParams) from a distribution.

    Matches the parameter ranges from ``HighwayTaskDistribution``:
      - vehicle_density: [0.5, 2.0]
      - reward_speed_range_low: [15.0, 25.0]
      - reward_speed_range_high: [25.0, 35.0]
      - collision_reward: [-2.0, -0.5]

    Args:
        key: PRNG key.
        base_params: base parameters to modify.
        vary_*: flags for which parameters to randomise.

    Returns:
        Modified EnvParams with sampled values.
    """
    k1, k2, k3, k4 = jrandom.split(key, 4)

    params = base_params

    if vary_vehicle_density:
        density = jrandom.uniform(k1, minval=0.5, maxval=2.0)
        params = params.replace(vehicles_density=density)

    if vary_speed_range:
        speed_low = jrandom.uniform(k2, minval=SPEED_RANGES["reward_speed_range_low"][0], maxval=SPEED_RANGES["reward_speed_range_low"][1])
        speed_high = jrandom.uniform(k3, minval=SPEED_RANGES["reward_speed_range_high"][0], maxval=SPEED_RANGES["reward_speed_range_high"][1])
        params = params.replace(
            reward_speed_range_low=speed_low,
            reward_speed_range_high=speed_high,
        )

    if vary_collision_reward:
        col_rew = jrandom.uniform(k4, minval=PENALTY_RANGES["collision_reward"][0], maxval=PENALTY_RANGES["collision_reward"][1])
        params = params.replace(collision_reward=col_rew)

    return params


def generate_task_batch(
    key: jnp.ndarray,
    base_params: EnvParams,
    n_tasks: int,
) -> EnvParams:
    """Generate a batch of task parameters (vmapped).

    Returns:
        EnvParams with all fields having an extra leading dimension
        of size ``n_tasks`` (pytree of (n_tasks, ...) arrays).
    """
    keys = jrandom.split(key, n_tasks)
    return jax.vmap(sample_task_params, in_axes=(0, None))(keys, base_params)


def collect_rollout(
    key: jnp.ndarray,
    state: EnvState,
    params: EnvParams,
    policy_fn: Callable,
    num_steps: int,
) -> Tuple[Dict[str, jnp.ndarray], EnvState]:
    """Collect a fixed-length rollout using ``jax.lax.scan``.

    Replaces the Python ``for`` loop in ``collect_rollout()`` from pearl_jax.py.
    Uses auto-reset so the scan can run for a fixed number of steps.

    Args:
        key: PRNG key.
        state: initial environment state.
        params: environment parameters.
        policy_fn: ``fn(key, obs) -> action`` — must be JIT-compatible.
        num_steps: number of steps to collect.

    Returns:
        (transitions, final_state) where transitions is a dict:
          - obs: (num_steps, obs_dim)
          - actions: (num_steps, action_dim)
          - rewards: (num_steps,)
          - next_obs: (num_steps, obs_dim)
          - terminals: (num_steps,)
    """

    def _step(carry, _):
        key, st = carry
        key, key_act, key_step = jrandom.split(key, 3)

        obs = observe(st, params)
        action = policy_fn(key_act, obs)

        next_obs, next_st, reward, done, info = HighwayJaxEnv.step_auto_reset(
            key_step, st, action, params,
        )

        # D4RL convention: terminals = true terminal (crash/goal),
        # timeouts = episode ended by time limit (truncation).
        crashed = info["crashed"].astype(jnp.float32)
        truncated = info["truncated"].astype(jnp.float32)

        transition = {
            "observations": obs,
            "actions": action,
            "rewards": reward,
            "next_observations": next_obs,
            "terminals": crashed,
            "timeouts": truncated,
            "truncated": truncated,  # Gymnasium-style alias for timeouts
        }

        return (key, next_st), transition

    (_, final_state), transitions = jax.lax.scan(
        _step, (key, state), None, length=num_steps,
    )

    return transitions, final_state


def collect_rollout_batched(
    key: jnp.ndarray,
    params_batch: EnvParams,
    policy_fn: Callable,
    num_steps: int,
    n_tasks: int,
) -> Dict[str, jnp.ndarray]:
    """Collect rollouts across multiple tasks in parallel via ``vmap``.

    Args:
        key: PRNG key.
        params_batch: batched EnvParams with leading dim = n_tasks.
        policy_fn: ``fn(key, obs) -> action``.
        num_steps: steps per task.
        n_tasks: number of tasks (must match params_batch leading dim).

    Returns:
        Dict with shape (n_tasks, num_steps, ...) for each field.

    Note:
        Currently supported only for HighwayJaxEnv. Other JAX env classes
        (MergeJaxEnv, RoundaboutJaxEnv, etc.) are not yet covered.
        TODO: generalize to all JAX env types.
    """
    keys = jrandom.split(key, n_tasks)

    def _collect_one(key_i, params_i):
        key_reset, key_collect = jrandom.split(key_i)
        obs, state = HighwayJaxEnv.reset(key_reset, params_i)
        transitions, _ = collect_rollout(
            key_collect, state, params_i, policy_fn, num_steps,
        )
        return transitions

    all_transitions = jax.vmap(_collect_one)(keys, params_batch)
    return all_transitions


# ------------------------------------------------------------------
#  JAX-native collect for pearl_jax.py integration
# ------------------------------------------------------------------

def jax_collect_data_for_task(
    key: jnp.ndarray,
    params: EnvParams,
    policy_fn: Callable,
    n_steps: int,
    max_path_length: int,
) -> Dict[str, jnp.ndarray]:
    """Collect data for a single task, returning padded arrays.

    This is the JAX replacement for ``collect_data_for_task`` in
    pearl_jax.py.  It runs a single rollout of ``n_steps`` total
    environment steps with auto-reset.

    Args:
        key: PRNG key.
        params: task-specific EnvParams.
        policy_fn: ``fn(key, obs) -> action``.
        n_steps: total environment steps to collect.
        max_path_length: maximum episode length (for auto-reset).

    Returns:
        Dict with (n_steps, ...) arrays for obs, actions, rewards,
        next_obs, terminals.
    """
    key_reset, key_collect = jrandom.split(key)
    _, state = HighwayJaxEnv.reset(key_reset, params)
    transitions, _ = collect_rollout(
        key_collect, state, params, policy_fn, n_steps,
    )
    return transitions
