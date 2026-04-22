from __future__ import annotations

from typing import Any

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.causal_mixin import CausalInterventionMixin
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork


class TwoWayEnv(CausalInterventionMixin, AbstractEnv):
    """
    A risk management task: the agent is driving on a two-way lane with icoming traffic.

    It must balance making progress by overtaking and ensuring safety.

    These conflicting objectives are implemented by a reward signal and a constraint signal,
    in the CMDP/BMDP framework.

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
    #  left_lane_reward        MID-EPISODE    reward weight in config
    #  high_speed_reward       MID-EPISODE    reward weight in config
    #  left_lane_constraint    MID-EPISODE    constraint weight in config
    #  other_vehicles_type     EPISODE-LEVEL  traffic vehicle class path

    _MID_EPISODE_VARS: set[str] = {
        "driver_aggressiveness",
        "speed_limit",
        "collision_reward",
        "left_lane_reward",
        "high_speed_reward",
        "left_lane_constraint",
    }

    _EPISODE_LEVEL_VARS: set[str] = {
        "other_vehicles_type",
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
                "collision_reward": 0,
                "left_lane_constraint": 1,
                "left_lane_reward": 0.2,
                "high_speed_reward": 0.8,
                "driver_aggressiveness": None,  # float ∈ [0,1] or None
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        return sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )

    def _rewards(self, action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        # Use continuous speed scaling instead of discrete speed_index
        scaled_speed = utils.lmap(
            self.vehicle.speed, [0, self.vehicle.MAX_SPEED], [0, 1]
        )
        return {
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "left_lane_reward": (
                len(neighbours) - 1 - self.vehicle.lane_index[2]
            )
            / max(len(neighbours) - 1, 1),
        }

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)

        # Reward decomposition for causal analysis
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
        """Push a mid-episode intervention into the live simulation."""
        super()._apply_mid_episode_intervention(variable, value)

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=800):
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        net = RoadNetwork()

        # Lanes
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 0],
                [length, 0],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, StraightLane.DEFAULT_WIDTH],
                [length, StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )
        net.add_lane(
            "b",
            "a",
            StraightLane(
                [length, 0], [0, 0], line_types=(LineType.NONE, LineType.NONE)
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the road

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(
                    road,
                    position=road.network.get_lane(("a", "b", 1)).position(
                        70 + 40 * i + 10 * self.np_random.normal(), 0
                    ),
                    heading=road.network.get_lane(("a", "b", 1)).heading_at(
                        70 + 40 * i
                    ),
                    speed=24 + 2 * self.np_random.normal(),
                    enable_lane_change=False,
                )
            )
        for i in range(2):
            v = vehicles_type(
                road,
                position=road.network.get_lane(("b", "a", 0)).position(
                    200 + 100 * i + 10 * self.np_random.normal(), 0
                ),
                heading=road.network.get_lane(("b", "a", 0)).heading_at(200 + 100 * i),
                speed=20 + 5 * self.np_random.normal(),
                enable_lane_change=False,
            )
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)
