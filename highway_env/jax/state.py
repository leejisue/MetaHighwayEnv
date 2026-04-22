"""Data structures for JAX highway environment.

Uses ``flax.struct.dataclass`` so that all structs are registered as
JAX pytrees automatically — compatible with ``jax.jit``, ``jax.vmap``,
``jax.lax.scan``, and ``jax.tree_util``.
"""

from __future__ import annotations

import flax.struct
import jax.numpy as jnp

# Maximum number of vehicles (ego + others).  JAX JIT requires static
# shapes, so we fix this and use an ``active`` mask for the actual count.
N_MAX = 50

# Compile-time constants for values that control array shapes or loop
# lengths inside JIT.  These CANNOT be part of EnvParams because
# flax.struct fields become traced values.
N_OBS = 5               # number of vehicles in the observation
N_SIM_PER_POLICY = 15   # simulation sub-steps per policy step (sim_freq // policy_freq)

# ---------- Intersection constants ----------
N_OBS_INTERSECTION = 15       # 15 observed vehicles
N_FEATURES_INTERSECTION = 7   # presence, x, y, vx, vy, cos_h, sin_h

# FIX P1-1 FINAL: N_LANES_MAX is a compile-time constant so that lane-clipping
# arithmetic inside JIT (e.g. jnp.clip(lane, 0, N_LANES_MAX - 1)) uses a
# Python-int upper bound known at trace time, preventing JIT recompilation when
# different EnvParams.num_lanes values are seen across meta-RL tasks.
# Used by HighwayJaxEnv + MergeJaxEnv clipping. Intersection/Roundabout/Racetrack have their own lane-limit constants.
N_LANES_MAX = 4               # maximum number of lanes across all highway tasks

# ---------- Racetrack constants ----------
N_GRID = 12                   # OccupancyGrid side length (12×12)
N_GRID_FEATURES = 2           # presence, on_road
N_TRACK_WAYPOINTS = 60        # fixed number of track waypoints for JIT
N_RACETRACK_SIM_PER_POLICY = 3  # racetrack: sim_freq(15) // policy_freq(5) = 3


@flax.struct.dataclass
class VehicleState:
    """Batched state of all vehicles on the road.

    Every field has shape ``(N_MAX,)``.  Inactive vehicles (beyond
    the current count) have ``active[i] == False``.
    """

    x: jnp.ndarray        # position x  (longitudinal)
    y: jnp.ndarray        # position y  (lateral)
    vx: jnp.ndarray       # speed along heading direction
    vy: jnp.ndarray       # lateral speed (usually 0 for kinematic model)
    heading: jnp.ndarray  # heading angle [rad]
    active: jnp.ndarray   # bool mask — True for live vehicles
    lane_id: jnp.ndarray  # current lane index (int)
    target_lane_id: jnp.ndarray  # target lane for lane-change (int)
    length: jnp.ndarray   # vehicle length (m)
    width: jnp.ndarray    # vehicle width (m)
    target_speed: jnp.ndarray  # desired cruising speed (m/s)


@flax.struct.dataclass
class EnvState:
    """Full environment state (immutable, JAX pytree)."""

    vehicles: VehicleState
    time: jnp.ndarray       # scalar, simulation time [s]
    step_count: jnp.ndarray  # scalar int, policy steps taken
    crashed: jnp.ndarray    # scalar bool
    truncated: jnp.ndarray  # scalar bool — True when step_count >= max_steps
    done: jnp.ndarray       # scalar bool — crashed | truncated


@flax.struct.dataclass
class EnvParams:
    """Environment parameters — varied per task in meta-RL.

    Default values match the original ``HighwayEnv.default_config()``.
    """

    # Road geometry
    # FIX P1-1: num_lanes is a Python int field in flax.struct — it is a
    # compile-time static value, NOT a traced JAX array.  Any task with a
    # different num_lanes value triggers a full JIT recompile.  In meta-RL,
    # keep num_lanes fixed across all tasks in a vmapped batch, or use a
    # module-level N_LANES_MAX constant and clip to it with jnp arithmetic
    # (e.g. jnp.clip(lane, 0, N_LANES_MAX - 1)) to avoid silent JIT thrash.
    num_lanes: int = N_LANES_MAX  # STATIC — default lane upper bound = N_LANES_MAX
    lane_width: float = 4.0
    road_length: float = 1000.0

    # Ego vehicle
    ego_initial_speed: float = 25.0
    ego_spacing: float = 2.0

    # Traffic
    num_vehicles: int = 20
    vehicles_density: float = 1.0

    # IDM parameters
    idm_desired_speed: float = 30.0
    idm_time_gap: float = 1.5
    idm_min_gap: float = 5.0
    idm_delta: float = 4.0
    idm_comfort_accel: float = 3.0
    idm_comfort_decel: float = 5.0
    idm_max_accel: float = 6.0

    # MOBIL parameters
    mobil_politeness: float = 0.0
    mobil_lane_change_min_acc_gain: float = 0.2
    mobil_lane_change_max_braking: float = 2.0
    mobil_lane_change_delay: float = 1.0

    # Reward weights
    collision_reward: float = -1.0
    high_speed_reward: float = 0.4
    right_lane_reward: float = 0.1
    lane_change_reward: float = 0.0
    reward_speed_range_low: float = 20.0
    reward_speed_range_high: float = 30.0
    normalize_reward: bool = True

    # Simulation timing
    # NOTE: simulation_frequency / policy_frequency are NOT here because
    # lax.scan needs a static length.  Use N_SIM_PER_POLICY constant instead.
    dt: float = 1.0 / 15.0           # simulation timestep (15 Hz)
    max_steps: int = 40               # policy steps per episode

    # NOTE: n_obs_vehicles is NOT here — use N_OBS constant for JIT shapes.

    # Vehicle dimensions (defaults)
    vehicle_length: float = 5.0
    vehicle_width: float = 2.0
    max_speed: float = 40.0


# =====================================================================
#  Merge environment params
# =====================================================================

@flax.struct.dataclass
class MergeEnvParams:
    """Parameters for the merge environment.

    Extends highway params with merging reward component.
    """

    # Road geometry (N_LANES_MAX main lanes + 1 merge lane)
    num_lanes: int = N_LANES_MAX  # default lane upper bound = N_LANES_MAX
    lane_width: float = 4.0
    road_length: float = 1000.0

    # Merge zone: vehicles in x > merge_start with lane_id >= num_lanes
    merge_start_x: float = 100.0
    merge_end_x: float = 250.0

    # Ego vehicle
    ego_initial_speed: float = 25.0
    ego_spacing: float = 2.0

    # Traffic
    num_vehicles: int = 20
    vehicles_density: float = 1.0
    num_merge_vehicles: int = 5

    # IDM parameters
    idm_desired_speed: float = 30.0
    idm_time_gap: float = 1.5
    idm_min_gap: float = 5.0
    idm_delta: float = 4.0
    idm_comfort_accel: float = 3.0
    idm_comfort_decel: float = 5.0
    idm_max_accel: float = 6.0

    # MOBIL parameters
    mobil_politeness: float = 0.0
    mobil_lane_change_min_acc_gain: float = 0.2
    mobil_lane_change_max_braking: float = 2.0
    mobil_lane_change_delay: float = 1.0

    # Reward weights
    collision_reward: float = -1.0
    high_speed_reward: float = 0.2
    right_lane_reward: float = 0.1
    lane_change_reward: float = -0.05
    merging_speed_reward: float = -0.5
    reward_speed_range_low: float = 20.0
    reward_speed_range_high: float = 30.0
    normalize_reward: bool = True

    dt: float = 1.0 / 15.0
    max_steps: int = 40

    vehicle_length: float = 5.0
    vehicle_width: float = 2.0
    max_speed: float = 40.0


# =====================================================================
#  Intersection environment params & state
# =====================================================================

@flax.struct.dataclass
class IntersectionEnvParams:
    """Parameters for the intersection environment."""

    # Intersection geometry
    intersection_size: float = 20.0  # half-width of intersection area
    road_length: float = 100.0       # length of each approach road
    lane_width: float = 4.0
    num_lanes_per_road: int = 1

    # Ego
    ego_initial_speed: float = 8.0
    ego_direction: int = -1  # -1=random, 0=S→N, 1=W→E, 2=N→S, 3=E→W

    # Traffic
    num_vehicles: int = 15
    vehicles_density: float = 1.0

    # IDM
    idm_desired_speed: float = 10.0
    idm_time_gap: float = 1.5
    idm_min_gap: float = 5.0
    idm_delta: float = 4.0
    idm_comfort_accel: float = 3.0
    idm_comfort_decel: float = 5.0
    idm_max_accel: float = 6.0

    # Reward
    collision_reward: float = -5.0
    high_speed_reward: float = 1.0
    arrived_reward: float = 1.0
    reward_speed_range_low: float = 7.0
    reward_speed_range_high: float = 10.0
    normalize_reward: bool = False  # Gym default is False for intersection

    dt: float = 1.0 / 15.0
    max_steps: int = 40

    vehicle_length: float = 5.0
    vehicle_width: float = 2.0
    max_speed: float = 15.0


@flax.struct.dataclass
class IntersectionEnvState:
    """State for intersection environment."""

    vehicles: VehicleState
    time: jnp.ndarray
    step_count: jnp.ndarray
    crashed: jnp.ndarray
    truncated: jnp.ndarray
    done: jnp.ndarray
    arrived: jnp.ndarray  # scalar bool — ego reached destination
    ego_direction: jnp.ndarray  # scalar int — actual direction chosen at reset


# =====================================================================
#  Roundabout environment params
# =====================================================================

@flax.struct.dataclass
class RoundaboutEnvParams:
    """Parameters for the roundabout environment.

    Models the roundabout as a wrapped road (vehicles circulate).
    """

    # Road geometry (roundabout approximated as circular multi-lane road)
    num_lanes: int = 2
    lane_width: float = 4.0
    roundabout_radius: float = 20.0
    road_length: float = 126.0  # ~2*pi*20

    ego_initial_speed: float = 8.0
    ego_spacing: float = 2.0

    num_vehicles: int = 15
    vehicles_density: float = 1.0

    # IDM
    idm_desired_speed: float = 10.0
    idm_time_gap: float = 1.5
    idm_min_gap: float = 5.0
    idm_delta: float = 4.0
    idm_comfort_accel: float = 3.0
    idm_comfort_decel: float = 5.0
    idm_max_accel: float = 6.0

    # MOBIL
    mobil_politeness: float = 0.0
    mobil_lane_change_min_acc_gain: float = 0.2
    mobil_lane_change_max_braking: float = 2.0
    mobil_lane_change_delay: float = 1.0

    # Reward
    collision_reward: float = -1.0
    high_speed_reward: float = 0.2
    right_lane_reward: float = 0.0
    lane_change_reward: float = -0.05
    reward_speed_range_low: float = 8.0
    reward_speed_range_high: float = 12.0
    normalize_reward: bool = True

    dt: float = 1.0 / 15.0
    max_steps: int = 40

    vehicle_length: float = 5.0
    vehicle_width: float = 2.0
    max_speed: float = 15.0


# =====================================================================
#  Racetrack environment params & state
# =====================================================================

@flax.struct.dataclass
class RacetrackEnvParams:
    """Parameters for the racetrack environment."""

    # Track is defined by waypoints (compile-time arrays stored separately)
    road_width: float = 8.0
    num_vehicles: int = 2

    # IDM
    idm_desired_speed: float = 8.0
    idm_time_gap: float = 1.5
    idm_min_gap: float = 5.0
    idm_delta: float = 4.0
    idm_comfort_accel: float = 3.0
    idm_comfort_decel: float = 5.0
    idm_max_accel: float = 6.0

    # Reward
    collision_reward: float = -1.0
    lane_centering_cost: float = 4.0
    lane_centering_reward: float = 1.0
    action_reward: float = -0.3
    reward_speed_range_low: float = 5.0
    reward_speed_range_high: float = 10.0

    dt: float = 1.0 / 15.0
    max_steps: int = 300

    vehicle_length: float = 5.0
    vehicle_width: float = 2.0
    max_speed: float = 15.0

    # OccupancyGrid params
    grid_min: float = -18.0
    grid_max: float = 18.0
    grid_step: float = 3.0


@flax.struct.dataclass
class RacetrackEnvState:
    """State for racetrack environment."""

    vehicles: VehicleState
    time: jnp.ndarray
    step_count: jnp.ndarray
    crashed: jnp.ndarray
    truncated: jnp.ndarray
    done: jnp.ndarray
    off_road: jnp.ndarray  # scalar bool — ego left the track
