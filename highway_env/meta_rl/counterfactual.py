"""Counterfactual reasoning engine for causal meta-RL experiments.

This module provides utilities for *counterfactual replay*: saving the
full simulator state at a point in time, then replaying from that state
under a different intervention to compare factual vs. counterfactual
trajectories.

The key abstraction is :class:`CounterfactualEngine`, which wraps a
``HighwayEnv`` (or ``MetaHighwayEnv``) and provides:

- ``save_state(env)``  →  opaque ``SimulatorState`` snapshot
- ``restore_state(env, state)``  →  deterministic rollback
- ``counterfactual_rollout(env, state, interventions, policy, steps)``
  →  trajectory under do(X := x) starting from saved state

Implementation
~~~~~~~~~~~~~~
State serialisation uses :func:`copy.deepcopy` on the *unwrapped*
environment (which already has a ``__deepcopy__`` that skips the viewer).
The ``np_random`` RNG state is captured and restored so that stochastic
transitions are identical across factual/counterfactual branches when
the same seed lineage is desired.

Example::

    from highway_env.meta_rl.counterfactual import CounterfactualEngine

    engine = CounterfactualEngine()

    # Factual trajectory
    obs, info = env.reset()
    for t in range(10):
        obs, r, term, trunc, info = env.step(policy(obs))
        if t == 5:
            state_t5 = engine.save_state(env)     # checkpoint

    # Counterfactual: what if aggressiveness had been 0.1 at t=5?
    cf_traj = engine.counterfactual_rollout(
        env, state_t5,
        interventions={"driver_aggressiveness": 0.1},
        policy=policy, steps=20,
    )
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SimulatorState:
    """Opaque snapshot of full simulator state.

    Users should not inspect or modify the internals directly.
    Use :meth:`CounterfactualEngine.restore_state` to apply.
    """
    env_snapshot: Any  # deep-copied AbstractEnv
    rng_state: dict  # np_random bit-generator state
    config_snapshot: dict
    time: float
    steps: int
    controlled_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CounterfactualEngine:
    """Engine for saving, restoring, and replaying environment states.

    This enables counterfactual reasoning of the form:

        "Given that the world was in state S at time t,
         what *would have happened* if we had done do(X := x)?"

    The engine is stateless — it operates on environment objects passed
    as arguments.
    """

    # ------------------------------------------------------------------
    #  State save / restore
    # ------------------------------------------------------------------

    @staticmethod
    def save_state(env) -> SimulatorState:
        """Capture a full snapshot of the environment's current state.

        The snapshot includes vehicle positions/speeds, road geometry,
        RNG state, config, and simulation clock.  It is safe to call
        ``save_state`` multiple times; each snapshot is independent.

        Args:
            env: A ``HighwayEnv`` (unwrapped) or ``MetaHighwayEnv``.
                 If wrapped, the unwrapped base env is used.

        Returns:
            An opaque :class:`SimulatorState` that can be passed to
            :meth:`restore_state`.
        """
        base = _unwrap(env)
        snapshot = copy.deepcopy(base)

        # Capture RNG state separately for explicit control
        rng_state = {}
        if hasattr(base, "np_random"):
            rng_state["np_random"] = _get_rng_state(base.np_random)
        if hasattr(base, "road") and base.road is not None:
            if hasattr(base.road, "np_random"):
                rng_state["road_np_random"] = _get_rng_state(
                    base.road.np_random
                )

        # Record controlled-vehicle indices into road.vehicles so that
        # restore_state can unambiguously re-link the ego vehicle(s)
        # without fragile position-based matching (Fix B).
        controlled_indices: List[int] = []
        if hasattr(base, "road") and base.road is not None:
            road_vehicles = list(base.road.vehicles)
            for cv in getattr(base, "controlled_vehicles", []):
                try:
                    controlled_indices.append(road_vehicles.index(cv))
                except ValueError:
                    # ego not in road.vehicles (shouldn't happen) — skip
                    pass

        return SimulatorState(
            env_snapshot=snapshot,
            rng_state=rng_state,
            config_snapshot=copy.deepcopy(base.config),
            time=getattr(base, "time", 0),
            steps=getattr(base, "steps", 0),
            controlled_indices=controlled_indices,
            metadata={"saved_at_time": getattr(base, "time", 0)},
        )

    @staticmethod
    def restore_state(env, state: SimulatorState) -> None:
        """Restore an environment to a previously saved state.

        After this call, ``env`` is in the *exact* same state as when
        :meth:`save_state` was called (modulo viewer, which is not
        serialised).

        Args:
            env:   The environment to restore.
            state: A :class:`SimulatorState` from :meth:`save_state`.
        """
        base = _unwrap(env)
        src = state.env_snapshot

        # Single deepcopy of the source road -- O(1) restore.
        # All vehicles (controlled + traffic) are part of road.vehicles,
        # so cloning the road clones every vehicle in one shot and
        # preserves identity references between objects (Fix B).
        base.road = copy.deepcopy(src.road)
        base.config = copy.deepcopy(state.config_snapshot)
        base.time = state.time
        base.steps = state.steps
        base.done = getattr(src, "done", False)

        # Re-link controlled vehicles using the recorded indices into
        # road.vehicles, captured at save_state() time.  This is
        # unambiguous and avoids fragile position-based matching.
        controlled_indices = getattr(state, "controlled_indices", []) or []
        if controlled_indices:
            base.controlled_vehicles = [
                base.road.vehicles[i]
                for i in controlled_indices
                if 0 <= i < len(base.road.vehicles)
            ]
        else:
            # Fallback (legacy snapshots without indices): deepcopy from
            # snapshot's controlled_vehicles list directly.
            base.controlled_vehicles = copy.deepcopy(src.controlled_vehicles)

        # Restore RNG state
        if "np_random" in state.rng_state and hasattr(base, "np_random"):
            _set_rng_state(base.np_random, state.rng_state["np_random"])
        if (
            "road_np_random" in state.rng_state
            and hasattr(base, "road")
            and base.road is not None
            and hasattr(base.road, "np_random")
        ):
            _set_rng_state(
                base.road.np_random, state.rng_state["road_np_random"]
            )

        # Re-define spaces (observation/action types may hold env refs)
        if hasattr(base, "define_spaces"):
            base.define_spaces()

    # ------------------------------------------------------------------
    #  Counterfactual rollout
    # ------------------------------------------------------------------

    @staticmethod
    def counterfactual_rollout(
        env,
        state: SimulatorState,
        interventions: Dict[str, Any],
        policy: Callable[[np.ndarray], int],
        steps: int,
        *,
        restore_after: bool = True,
    ) -> Dict[str, Any]:
        """Rollout from a saved state under a counterfactual intervention.

        1. Saves the *current* env state (so we can restore later).
        2. Restores ``state`` (the checkpoint).
        3. Applies ``interventions`` via ``env.intervene()``.
        4. Steps ``policy`` for ``steps`` steps, collecting trajectory.
        5. Optionally restores the pre-rollout state.

        Args:
            env:            ``HighwayEnv`` or ``MetaHighwayEnv``.
            state:          Checkpoint from :meth:`save_state`.
            interventions:  ``{variable: value}`` for do-operator.
            policy:         ``obs → action`` callable.
            steps:          Number of environment steps to collect.
            restore_after:  If True, restore env to its state before
                            this call (default True).

        Returns:
            Dict with keys:

            - ``observations``: list of obs arrays
            - ``actions``: list of actions
            - ``rewards``: list of floats
            - ``terminated``: list of bools
            - ``truncated``: list of bools
            - ``infos``: list of info dicts
            - ``interventions``: the interventions applied
            - ``total_return``: sum of rewards
        """
        base = _unwrap(env)

        # Optionally save current state to restore later
        pre_state = None
        if restore_after:
            pre_state = CounterfactualEngine.save_state(env)

        # Restore to checkpoint
        CounterfactualEngine.restore_state(env, state)

        # Apply interventions
        intervention_meta = {}
        for var, val in interventions.items():
            if hasattr(base, "intervene"):
                intervention_meta[var] = base.intervene(var, val)

        # Rollout
        trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
        }

        # Fix C: observe AFTER all interventions have been applied so
        # that the first action sees post-intervention state (e.g.
        # mutated IDM aggressiveness already affects neighbouring
        # vehicles' kinematics encoded in the observation).
        obs = base.observation_type.observe()
        for _ in range(steps):
            action = policy(obs)
            result = base.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, done, info = result
                terminated = done
                truncated = False

            trajectory["observations"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["terminated"].append(terminated)
            trajectory["truncated"].append(truncated)
            trajectory["infos"].append(info)

            if terminated or truncated:
                break

        trajectory["interventions"] = interventions
        trajectory["intervention_meta"] = intervention_meta
        trajectory["total_return"] = sum(trajectory["rewards"])

        # Restore pre-rollout state if requested
        if restore_after and pre_state is not None:
            CounterfactualEngine.restore_state(env, pre_state)

        return trajectory

    # ------------------------------------------------------------------
    #  Convenience: factual vs. counterfactual comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_factual_counterfactual(
        env,
        state: SimulatorState,
        factual_policy: Callable[[np.ndarray], int],
        counterfactual_interventions: Dict[str, Any],
        counterfactual_policy: Optional[Callable[[np.ndarray], int]] = None,
        steps: int = 20,
    ) -> Dict[str, Any]:
        """Run both factual and counterfactual rollouts and compare.

        Args:
            env:    The environment.
            state:  Checkpoint state.
            factual_policy:  Policy for factual world.
            counterfactual_interventions:  do(X:=x) for counterfactual.
            counterfactual_policy:  Policy for counterfactual (defaults
                                    to factual_policy).
            steps:  Rollout length.

        Returns:
            Dict with ``factual``, ``counterfactual``, and ``delta``
            (difference in total returns).
        """
        if counterfactual_policy is None:
            counterfactual_policy = factual_policy

        # Factual rollout (no interventions)
        factual = CounterfactualEngine.counterfactual_rollout(
            env, state,
            interventions={},
            policy=factual_policy,
            steps=steps,
            restore_after=True,
        )

        # Counterfactual rollout
        counterfactual = CounterfactualEngine.counterfactual_rollout(
            env, state,
            interventions=counterfactual_interventions,
            policy=counterfactual_policy,
            steps=steps,
            restore_after=True,
        )

        return {
            "factual": factual,
            "counterfactual": counterfactual,
            "delta_return": (
                counterfactual["total_return"] - factual["total_return"]
            ),
        }


# ======================================================================
#  Internal helpers
# ======================================================================

def _unwrap(env):
    """Get the unwrapped base environment."""
    while hasattr(env, "unwrapped") and env.unwrapped is not env:
        env = env.unwrapped
    # MetaHighwayEnv stores base env differently
    if hasattr(env, "base_env"):
        return env.base_env
    return env


def _get_rng_state(rng) -> dict:
    """Extract RNG state from either numpy legacy or Generator."""
    if hasattr(rng, "bit_generator"):
        # numpy Generator (gymnasium style)
        return {"state": rng.bit_generator.state}
    elif hasattr(rng, "get_state"):
        # numpy legacy RandomState
        return {"state": rng.get_state()}
    return {}


def _set_rng_state(rng, state_dict: dict) -> None:
    """Restore RNG state."""
    if "state" not in state_dict:
        return
    if hasattr(rng, "bit_generator"):
        rng.bit_generator.state = state_dict["state"]
    elif hasattr(rng, "set_state"):
        rng.set_state(state_dict["state"])


def _vehicles_match(v1, v2) -> bool:
    """Check if two vehicles represent the same entity (by position)."""
    try:
        return (
            np.allclose(v1.position, v2.position, atol=0.01)
            and abs(v1.speed - v2.speed) < 0.01
        )
    except (AttributeError, TypeError):
        return False
