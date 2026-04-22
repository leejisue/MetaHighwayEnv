"""Unified Causal Intervention API mixin for all Highway-env environments.

This module provides :class:`CausalInterventionMixin`, a mixin class that
adds a consistent ``intervene()`` / ``intervene_batch()`` /
``interventionable_variables()`` / ``get_causal_state()`` /
``decompose_reward()`` API to any environment that inherits from
:class:`~highway_env.envs.common.abstract.AbstractEnv`.

Design principles
~~~~~~~~~~~~~~~~~
- **Shared logic lives in the mixin** — the ``intervene()`` dispatch loop,
  metadata construction, granularity classification, and
  ``intervene_batch()`` wrapper are all here.
- **Per-env specifics are declarative** — each environment subclass only
  needs to define ``_MID_EPISODE_VARS``, ``_EPISODE_LEVEL_VARS``, and
  override ``_apply_mid_episode_intervention()`` for any variable that
  needs live-object mutation (as opposed to a pure config update).
- **Backward compatible** — ``HighwayEnv`` already defined these class
  attributes and methods; the mixin extracts the common pattern so that
  the same API works on Intersection, Merge, Roundabout, and Racetrack.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from highway_env.vehicle.behavior import IDMVehicle


class CausalInterventionMixin:
    """Mixin that adds do-calculus style interventions to any AbstractEnv.

    Subclasses MUST define:

    - ``_MID_EPISODE_VARS: set[str]``  — config keys that can change mid-episode
    - ``_EPISODE_LEVEL_VARS: set[str]`` — config keys that require ``reset()``

    Subclasses SHOULD override:

    - ``_apply_mid_episode_intervention(variable, value)`` — for variables
      that need live-object mutation beyond a config update.
    - ``get_causal_state()`` — to return a structured dict of the env's
      current causal variables.
    - ``decompose_reward(info)`` — to return env-specific reward components.
    """

    # ---- Subclass must populate these (or inherit from their env) ----
    _MID_EPISODE_VARS: set[str] = set()
    _EPISODE_LEVEL_VARS: set[str] = set()

    # =================================================================
    #  intervene() — the core do(variable := value) operation
    # =================================================================

    def intervene(
        self,
        variable: str,
        value: Any,
    ) -> dict[str, Any]:
        """Perform a do-calculus style intervention: ``do(variable := value)``.

        Only the specified variable is modified; all other environment
        parameters remain at their current values.

        Args:
            variable: Name of the causal variable to intervene on.
            value:    The interventional value to set.

        Returns:
            Metadata dict with keys: ``variable``, ``value``,
            ``previous_value``, ``granularity``, ``requires_reset``,
            ``applied``.

        Raises:
            ValueError: If *variable* is not a recognised interventionable
                        variable for this environment.
        """
        all_vars = self._MID_EPISODE_VARS | self._EPISODE_LEVEL_VARS
        if variable not in all_vars:
            raise ValueError(
                f"Unknown intervention variable: {variable!r} for "
                f"{type(self).__name__}.  "
                f"Choose from: {sorted(all_vars)}"
            )

        previous_value = self.config.get(variable)

        # Always store in config so that future resets respect it.
        self.config[variable] = value

        if variable in self._MID_EPISODE_VARS:
            self._apply_mid_episode_intervention(variable, value)
            return {
                "variable": variable,
                "value": value,
                "previous_value": previous_value,
                "granularity": "mid_episode",
                "requires_reset": False,
                "applied": True,
            }
        else:
            # Episode-level: config updated but scene NOT rebuilt.
            return {
                "variable": variable,
                "value": value,
                "previous_value": previous_value,
                "granularity": "episode_level",
                "requires_reset": True,
                "applied": False,
            }

    # =================================================================
    #  _apply_mid_episode_intervention — default handlers
    # =================================================================

    def _apply_mid_episode_intervention(
        self, variable: str, value: Any
    ) -> None:
        """Push a mid-episode intervention into the live simulation.

        The default implementation handles variables that are common across
        all environments:

        - ``driver_aggressiveness`` → mutate all non-ego IDMVehicles
        - ``speed_limit`` → mutate all lane objects
        - Reward weights / ``duration`` → pure config (no-op here)

        Subclasses may override or ``super()`` + extend for env-specific
        variables.
        """
        if variable == "driver_aggressiveness":
            # Mutate every non-ego IDMVehicle currently on the road.
            if hasattr(self, "road") and self.road is not None:
                controlled = getattr(self, "controlled_vehicles", [])
                for v in self.road.vehicles:
                    if v not in controlled and isinstance(v, IDMVehicle):
                        v.set_aggressiveness(value)

        elif variable == "speed_limit":
            # Mutate every lane object in the road network.
            if hasattr(self, "road") and self.road is not None:
                for lane in self.road.network.lanes_list():
                    lane.speed_limit = float(value)

        # All other mid-episode vars (reward weights, duration, etc.) are
        # purely config-driven — updating self.config is sufficient.

    # =================================================================
    #  intervene_batch
    # =================================================================

    def intervene_batch(
        self,
        interventions: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Apply multiple interventions at once.

        Returns:
            ``{variable_name: metadata_dict, ...}``
        """
        return {var: self.intervene(var, val) for var, val in interventions.items()}

    # =================================================================
    #  interventionable_variables  (class method)
    # =================================================================

    @classmethod
    def interventionable_variables(cls) -> dict[str, str]:
        """Return ``{variable_name: granularity}`` for this environment.

        Useful for programmatic introspection.
        """
        out = {}
        for v in sorted(cls._MID_EPISODE_VARS):
            out[v] = "mid_episode"
        for v in sorted(cls._EPISODE_LEVEL_VARS):
            out[v] = "episode_level"
        return out

    # =================================================================
    #  get_causal_state — structured snapshot of causal variables
    # =================================================================

    def get_causal_state(self) -> dict[str, Any]:
        """Return a structured, serializable dict of the env's causal state.

        This includes all interventionable variables with their current
        values, plus derived state like ego vehicle speed and position.

        Subclasses should ``super()`` and extend.
        """
        state: dict[str, Any] = {}

        # All config-level causal variables
        for var in sorted(self._MID_EPISODE_VARS | self._EPISODE_LEVEL_VARS):
            state[var] = self.config.get(var)

        # Common derived state
        vehicle = getattr(self, "vehicle", None)
        if vehicle is not None:
            state["ego_speed"] = float(vehicle.speed)
            state["ego_crashed"] = bool(vehicle.crashed)
            state["ego_on_road"] = bool(getattr(vehicle, "on_road", True))
            pos = getattr(vehicle, "position", None)
            if pos is not None:
                state["ego_position"] = [float(x) for x in pos]

        # Traffic summary
        road = getattr(self, "road", None)
        if road is not None:
            controlled = set(id(v) for v in getattr(self, "controlled_vehicles", []))
            traffic_vehicles = [v for v in road.vehicles if id(v) not in controlled]
            state["traffic_count"] = len(traffic_vehicles)
            if traffic_vehicles:
                speeds = [v.speed for v in traffic_vehicles]
                state["traffic_mean_speed"] = float(np.mean(speeds))

        return state

    # =================================================================
    #  decompose_reward — structured reward breakdown
    # =================================================================

    def decompose_reward(self, info: dict | None = None) -> dict[str, dict[str, float]]:
        """Return decomposed reward components from the last step.

        This wraps the existing ``info["reward_components"]`` when it is
        present (HighwayEnv), or constructs it from ``info["rewards"]``
        and config weights for other environments.

        Args:
            info: The info dict returned by ``step()``.  If None, the
                  method attempts to read from the last call's info.

        Returns:
            ``{component_name: {"raw": float, "weight": float, "weighted": float}}``
        """
        if info is None:
            info = {}

        # Already decomposed (HighwayEnv does this in _info)
        if "reward_components" in info:
            return info["reward_components"]

        # Build from raw rewards dict
        raw = info.get("rewards", {})
        components = {}
        for name, value in raw.items():
            weight = self.config.get(name, 0)
            components[name] = {
                "raw": float(value),
                "weight": float(weight),
                "weighted": float(value * weight),
            }
        return components
