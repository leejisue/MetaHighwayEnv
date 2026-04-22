"""JAX-native highway environments for GPU-accelerated meta-RL training.

Public API
----------
    HighwayJaxEnv        — highway env (straight multi-lane road)
    MergeJaxEnv          — merge env (highway + merge lane)
    IntersectionJaxEnv   — 4-way intersection env
    RoundaboutJaxEnv     — roundabout (wrapped circular road)
    RacetrackJaxEnv      — oval racetrack with OccupancyGrid obs

    EnvState / EnvParams — highway / merge / roundabout state & params
    IntersectionEnvState / IntersectionEnvParams
    RacetrackEnvState    / RacetrackEnvParams
    MergeEnvParams / RoundaboutEnvParams

    VehicleState         — batched vehicle states
    N_MAX                — maximum number of vehicles (compile-time constant)

Action Spaces (matching Highway-Env)
-------------------------------------
Each JAX env provides both ``step(action_2d)`` (raw continuous) and, for
discrete envs, ``step_discrete(action_int)`` / ``step_auto_reset_discrete()``.

**Use ``step_discrete`` for discrete envs** — it matches Highway-Env behavior.

  ===============  ============  =====  ===================================
  Environment      Action Type   Dim    Details
  ===============  ============  =====  ===================================
  highway          Discrete      5      0=LEFT 1=IDLE 2=RIGHT 3=FAST 4=SLOW
  merge            Discrete      5      same as highway
  roundabout       Discrete      5      same; speeds=[0, 8, 16]
  intersection     Discrete      3      0=SLOWER 1=IDLE 2=FASTER (no lane)
  racetrack        Continuous    2      [accel, steering] in [-1, 1]
  ===============  ============  =====  ===================================

Target speeds per environment:
  - highway / merge: [20, 25, 30] m/s
  - roundabout:      [0, 8, 16] m/s
  - intersection:    [0, 4.5, 9] m/s

For training:
  - Discrete envs → categorical policy (SAC-Discrete), see ``jax_sac.py``
  - Continuous envs → TanhGaussian policy (standard SAC)
"""

from .state import (
    VehicleState, EnvState, EnvParams, N_MAX, N_OBS, N_SIM_PER_POLICY,
    N_LANES_MAX,
    MergeEnvParams,
    IntersectionEnvParams, IntersectionEnvState,
    RoundaboutEnvParams,
    RacetrackEnvParams, RacetrackEnvState,
)
from .env import HighwayJaxEnv
from .env_merge import MergeJaxEnv
from .env_intersection import IntersectionJaxEnv
from .env_roundabout import RoundaboutJaxEnv
from .env_racetrack import RacetrackJaxEnv
from .observation import observe, obs_shape
from .reward import compute_reward
from .discrete_action import (
    N_DISCRETE_ACTIONS,
    N_DISCRETE_ACTIONS_LONGI,
    HIGHWAY_TARGET_SPEEDS,
    MERGE_TARGET_SPEEDS,
    ROUNDABOUT_TARGET_SPEEDS,
    INTERSECTION_TARGET_SPEEDS,
)
from .meta_env import (
    sample_task_params,
    generate_task_batch,
    collect_rollout,
    collect_rollout_batched,
    jax_collect_data_for_task,
)

__all__ = [
    # Env classes
    "HighwayJaxEnv",
    "MergeJaxEnv",
    "IntersectionJaxEnv",
    "RoundaboutJaxEnv",
    "RacetrackJaxEnv",
    # State / params
    "EnvState",
    "EnvParams",
    "MergeEnvParams",
    "IntersectionEnvParams",
    "IntersectionEnvState",
    "RoundaboutEnvParams",
    "RacetrackEnvParams",
    "RacetrackEnvState",
    "VehicleState",
    # Constants
    "N_MAX",
    "N_LANES_MAX",
    "N_OBS",
    "N_SIM_PER_POLICY",
    # Highway-specific helpers
    "observe",
    "obs_shape",
    "compute_reward",
    "sample_task_params",
    "generate_task_batch",
    "collect_rollout",
    "collect_rollout_batched",
    "jax_collect_data_for_task",
]
