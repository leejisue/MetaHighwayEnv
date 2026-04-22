"""Centralized parameter ranges for Meta-RL task distributions.

Single source of truth — referenced by task_distribution, env_factory, README.
"""

# Density per scenario (vehicles/lane or count)
DENSITY_RANGES = {
    "highway":      (0.5, 3.0),
    "merge":        (0.5, 3.0),
    "intersection": (0.5, 3.0),
    "roundabout":   (0.5, 3.0),
    "racetrack":    (0.5, 3.0),
}

# Speed reward range per scenario [low, high] in m/s
SPEED_RANGES = {
    "reward_speed_range_low":  (15.0, 25.0),
    "reward_speed_range_high": (25.0, 35.0),
}

# Collision penalty range (negative values)
PENALTY_RANGES = {
    "collision_reward": (-2.0, -0.5),
}

# Reward weight ranges for per-task reward customisation
REWARD_PARAM_RANGES = {
    "collision_reward":   (-2.0, -0.5),
    "high_speed_reward":  (0.1,  0.8),
    "right_lane_reward":  (0.0,  0.3),
    "lane_change_reward": (-0.2, 0.0),
}
