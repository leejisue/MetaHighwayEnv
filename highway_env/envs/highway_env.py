from __future__ import annotations

from typing import Any

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.causal_mixin import CausalInterventionMixin
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(CausalInterventionMixin, AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.

    Includes the full Causal Intervention API via :class:`CausalInterventionMixin`.
    """

    # =================================================================
    #  Causal Intervention API — variable declarations
    # =================================================================
    #
    #  Variable name           Granularity    What changes
    #  ──────────────────────  ─────────────  ─────────────────────────
    #  driver_aggressiveness   MID-EPISODE    IDM/MOBIL params of every
    #                                         non-ego IDMVehicle on road
    #  speed_limit             MID-EPISODE    lane.speed_limit for all
    #                                         lanes in road.network
    #  collision_reward        MID-EPISODE    reward weight in config
    #  high_speed_reward       MID-EPISODE    reward weight in config
    #  right_lane_reward       MID-EPISODE    reward weight in config
    #  lane_change_reward      MID-EPISODE    reward weight in config
    #  reward_speed_range      MID-EPISODE    [low, high] speed→reward
    #  duration                MID-EPISODE    episode time limit
    #  vehicles_count          EPISODE-LEVEL  number of spawned vehicles
    #  vehicles_density        EPISODE-LEVEL  spacing between vehicles
    #  lanes_count             EPISODE-LEVEL  road lane count

    _MID_EPISODE_VARS: set[str] = {
        "driver_aggressiveness",
        "speed_limit",
        "collision_reward",
        "high_speed_reward",
        "right_lane_reward",
        "lane_change_reward",
        "reward_speed_range",
        "duration",
    }

    _EPISODE_LEVEL_VARS: set[str] = {
        "vehicles_count",
        "vehicles_density",
        "lanes_count",
    }

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": True,
                "driver_aggressiveness": None,  # float ∈ [0,1] or None (default IDM)
                "speed_limit": 30,  # default lane speed limit (m/s)
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"],
                speed_limit=self.config.get("speed_limit", 30),
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                # Apply driver aggressiveness to IDM parameters
                agg = self.config.get("driver_aggressiveness")
                if agg is not None and isinstance(vehicle, IDMVehicle):
                    vehicle.set_aggressiveness(agg)
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        # Detect lane change: action 0 = LANE_LEFT, action 2 = LANE_RIGHT
        is_lane_change = float(action in (0, 2)) if isinstance(action, (int, np.integer)) else 0.0
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "lane_change_reward": is_lane_change,
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        """
        Return info dict with decomposed reward components.

        Adds ``info["reward_components"]`` containing the *weighted* contribution
        of each reward signal (speed, collision, right_lane, lane_change) so that
        causal-analysis code can inspect which component drove the total reward.
        """
        info = super()._info(obs, action)

        # Build reward_components from the raw (unweighted) reward dict that
        # AbstractEnv._info already stored in info["rewards"]
        raw = info.get("rewards", {})
        components = {}
        for name, value in raw.items():
            weight = self.config.get(name, 0)
            components[name] = {
                "raw": float(value),
                "weight": float(weight),
                "weighted": float(value * weight),
            }
        info["reward_components"] = components

        # Expose driver_aggressiveness in info for causal tracing
        agg = self.config.get("driver_aggressiveness")
        if agg is not None:
            info["driver_aggressiveness"] = float(agg)

        return info

    # =================================================================
    #  Override mixin: env-specific mid-episode handlers
    # =================================================================

    def _apply_mid_episode_intervention(
        self, variable: str, value: Any
    ) -> None:
        """Push a mid-episode intervention into the live simulation.

        Called by :meth:`CausalInterventionMixin.intervene` for variables
        in ``_MID_EPISODE_VARS``.
        """
        # Delegate common handlers (driver_aggressiveness, speed_limit)
        # to the mixin, which handles them generically.
        super()._apply_mid_episode_intervention(variable, value)

        # Highway-specific: no additional handlers needed beyond the
        # mixin's defaults.  Reward weights and duration are purely
        # config-driven.

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or off-road."""
        return (
            self.vehicle.crashed
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
