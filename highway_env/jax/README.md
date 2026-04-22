# MetaHighway JAX Environments

Pure-functional JAX implementations of highway-env environments for meta-RL research.
All environments follow the gymnax pattern and are compatible with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

## Environment Specifications

### Action Space

All JAX environments use **continuous 2D actions** `(acceleration, steering)` for compatibility
with continuous-action RL algorithms (SAC, PEARL). This differs from highway-env's default
discrete meta-actions for some environments.

| Environment | JAX Action | highway-env Default | highway-env ContinuousAction | Notes |
|---|---|---|---|---|
| **highway** | `(2,)` [accel, steer] | DiscreteMetaAction, Discrete(5) | Box(2,) [accel, steer] | JAX uses continuous equivalent |
| **merge** | `(2,)` [accel, steer] | DiscreteMetaAction, Discrete(5) | Box(2,) [accel, steer] | JAX uses continuous equivalent |
| **roundabout** | `(2,)` [accel, steer] | DiscreteMetaAction, Discrete(5) | Box(2,) [accel, steer] | JAX uses continuous equivalent |
| **intersection** | `(2,)` [accel, steer] | DiscreteMetaAction, Discrete(3) | Box(2,) [accel, steer] | Original: longitudinal only |
| **parking** | `(2,)` [accel, steer] | ContinuousAction, Box(2,) | Box(2,) [accel, steer] | Matches original |
| **racetrack** | `(2,)` [accel, steer] | ContinuousAction, Box(1,) | Box(1,) [steer] | accel zeroed internally |

**Action semantics:**
- `action[0]`: acceleration (m/s^2). Racetrack ignores this (steering-only control).
- `action[1]`: steering angle (rad). Racetrack clips to [-pi/4, pi/4].

**highway-env ContinuousAction mapping** (for reference):
- acceleration: [-1, 1] -> [-5.0, 5.0] m/s^2
- steering: [-1, 1] -> [-pi/4, pi/4] rad

The JAX environments accept **raw values** (not normalized [-1,1]). Per-env clipping:

| Environment | Accel Clipping | Steering Clipping |
|---|---|---|
| highway, merge, roundabout | No clipping (applied directly) | No clipping (applied directly) |
| intersection | No clipping | No clipping |
| parking | Speed clipped to [-max_speed, max_speed] | No clipping |
| racetrack | Ignored (set to 0) | Clipped to [-pi/4, pi/4] |

If your policy outputs normalized [-1,1] actions, scale them before passing to the environment.

### Observation Space

| Environment | Obs Dim | Structure | Features |
|---|---|---|---|
| **highway** | 25 | (5 vehicles x 5 features) | [presence, x, y, vx, vy] ego-relative, normalized |
| **merge** | 25 | (5 vehicles x 5 features) | [presence, x, y, vx, vy] ego-relative, normalized |
| **roundabout** | 25 | (5 vehicles x 5 features) | [presence, x, y, vx, vy] **absolute** coords, normalized |
| **intersection** | 105 | (15 vehicles x 7 features) | [presence, x, y, vx, vy, cos_h, sin_h] absolute, normalized |
| **parking** | 18 | 3 x 6 features | [ego(6), achieved_goal(6), desired_goal(6)] each: [x/100, y/100, vx/5, vy/5, cos_h, sin_h] |
| **racetrack** | 288 | (2 features x 12 x 12 grid) | OccupancyGrid: [presence, on_road] ego-centered |

### Info Dict

All environments return a dict with at least `crashed` and `truncated` keys.

| Environment | Info Keys | Notes |
|---|---|---|
| **highway** | `crashed`, `truncated`, `step_count` | |
| **merge** | `crashed`, `truncated`, `step_count` | |
| **roundabout** | `crashed`, `truncated`, `step_count` | |
| **intersection** | `crashed`, `truncated`, `arrived`, `step_count` | `arrived`: ego reached destination |
| **parking** | `crashed`, `truncated`, `success`, `out_of_bounds`, `weighted_dist`, `step_count` | `crashed` = `out_of_bounds` (no traffic); terminates on success, out_of_bounds, or truncation |
| **racetrack** | `crashed`, `truncated`, `off_road`, `step_count` | `off_road`: ego left track |

**Note on parking `crashed`:** The JAX parking env has no traffic vehicles, so there are
no vehicle-vehicle collisions. `info["crashed"]` is set to `info["out_of_bounds"]`
(boundary violation) for API consistency with other environments.

### Parameters

| Environment | Params Class | Key Parameters |
|---|---|---|
| **highway** | `EnvParams` | `vehicles_density`, `collision_reward`, `reward_speed_range_*`, `max_steps=40` |
| **merge** | `MergeEnvParams` | `merge_start_x`, `merge_end_x`, `num_merge_vehicles`, `max_steps=40` |
| **roundabout** | `RoundaboutEnvParams` | `roundabout_radius`, `road_length=126`, `max_steps=40` |
| **intersection** | `IntersectionEnvParams` | `intersection_size`, `ego_direction`, `arrived_reward`, `max_steps=40` |
| **parking** | `ParkingEnvParams` | `goal_x/y/heading`, `reward_weight_*`, `lot_length/width`, `max_steps=100` |
| **racetrack** | `RacetrackEnvParams` | `road_width`, `lane_centering_*`, `grid_*`, `max_steps=300` |

### State Classes

| Environment | State Class | Shared with |
|---|---|---|
| **highway** | `EnvState` | merge, roundabout |
| **merge** | `EnvState` | highway, roundabout |
| **roundabout** | `EnvState` | highway, merge |
| **intersection** | `IntersectionEnvState` | (unique: adds `arrived`, `ego_direction`) |
| **parking** | `ParkingEnvState` | (unique: ego-only, no `VehicleState`) |
| **racetrack** | `RacetrackEnvState` | (unique: adds `off_road`) |

## API

All environments follow the gymnax pattern with static methods:

```python
import jax
import jax.random as jrandom
from highway_env.jax.env import HighwayJaxEnv
from highway_env.jax.state import EnvParams

# Create parameters
params = EnvParams()

# Reset
key = jrandom.PRNGKey(0)
obs, state = HighwayJaxEnv.reset(key, params)
# obs: (25,), state: EnvState

# Step
key, key_step = jrandom.split(key)
action = jnp.array([1.0, 0.0])  # accelerate, no steering
obs, state, reward, done, info = HighwayJaxEnv.step(key_step, state, action, params)

# Step with auto-reset (for rollout collection)
obs, state, reward, done, info = HighwayJaxEnv.step_auto_reset(key_step, state, action, params)
```

### Collecting Rollouts with `jax.lax.scan`

```python
from highway_env.jax.meta_env import collect_rollout

def random_policy(key, obs):
    return jrandom.uniform(key, shape=(2,), minval=-1.0, maxval=1.0)

key = jrandom.PRNGKey(42)
key_reset, key_collect = jrandom.split(key)

params = EnvParams()
obs, state = HighwayJaxEnv.reset(key_reset, params)

transitions, final_state = collect_rollout(
    key_collect, state, params, random_policy, num_steps=1000
)
# transitions["observations"]:      (1000, 25)
# transitions["actions"]:           (1000, 2)
# transitions["rewards"]:           (1000,)
# transitions["next_observations"]: (1000, 25)
# transitions["terminals"]:         (1000,)
# transitions["truncated"]:         (1000,)
```

### Using Different Environments

```python
from highway_env.jax.env_parking import ParkingJaxEnv
from highway_env.jax.state import ParkingEnvParams

params = ParkingEnvParams(goal_x=10.0, goal_y=-8.0, goal_heading=0.0)
obs, state = ParkingJaxEnv.reset(key, params)
# obs: (18,) = [ego(6), achieved_goal(6), desired_goal(6)]

action = jnp.array([0.5, 0.1])  # gentle acceleration + slight steering
obs, state, reward, done, info = ParkingJaxEnv.step(key, state, action, params)
print(info["success"])  # True if reached goal
```

## Differences from highway-env

| Aspect | highway-env | MetaHighway JAX |
|---|---|---|
| **Action type** | Configurable (discrete/continuous) | Always continuous (2,) |
| **Action range** | Normalized [-1, 1] for continuous | Raw values (no normalization) |
| **State** | Mutable Python objects | Immutable flax.struct dataclass |
| **Execution** | Sequential Python | JIT-compiled, vmap-compatible |
| **Traffic model** | Full IDM + MOBIL | Simplified (env-dependent) |
| **Rendering** | Pygame visualization | No rendering (data collection only) |
| **Racetrack action** | 1D steering only | 2D but accel ignored internally |
| **Intersection action** | 3 discrete (speed only) | 2D continuous (accel + steering) |
| **Parking crash** | Vehicle collision detection | Out-of-bounds only (no traffic) |

## Offline Datasets

### Dataset Schema (D4RL + Gymnasium)

Each `.npz` file contains:

| Key | Shape | Type | Description |
|---|---|---|---|
| `observations` | `(T, obs_dim)` | float32 | States |
| `actions` | `(T, 2)` | float32 | [acceleration, steering] |
| `rewards` | `(T,)` | float32 | Per-step rewards |
| `next_observations` | `(T, obs_dim)` | float32 | Next states |
| `terminals` | `(T,)` | float32 | 1.0 if natural termination (crash/success) |
| `timeouts` | `(T,)` | float32 | 1.0 if truncated by max_steps (D4RL convention) |
| `truncated` | `(T,)` | float32 | Same as `timeouts` (Gymnasium convention alias) |
| `obs_mean` | `(obs_dim,)` | float32 | Observation mean for normalization |
| `obs_std` | `(obs_dim,)` | float32 | Observation std for normalization |
| `act_mean` | `(2,)` | float32 | Action mean |
| `act_std` | `(2,)` | float32 | Action std |
| `episode_returns` | `(n_episodes,)` | float32 | Per-episode cumulative rewards |
| `episode_lengths` | `(n_episodes,)` | int32 | Per-episode step counts |
| `env_name` | scalar | str | Environment type (e.g., "highway") |
| `task_name` | scalar | str | Task variant (e.g., "easy") |
| `level` | scalar | str | Policy quality ("random", "medium", "expert") |
| `seed` | scalar | int | Random seed used |

**Key distinction:**
- `terminals` (D4RL) = natural episode end (crash, out-of-bounds, goal reached)
- `truncated` / `timeouts` = episode ended by `max_steps` limit
- Both should NOT be 1.0 simultaneously for the same transition

### Storage Layout

```
benchmarks/datasets/
  meta_highway/          # 6 task variants x 3 levels = 18 files
    easy_random_seed0.npz
    easy_medium_seed0.npz
    easy_expert_seed0.npz
    dense_random_seed0.npz
    ...
  meta_intersection/     # 6 task variants
    default_random_seed0.npz
    dense_random_seed0.npz
    ...
  meta_merge/            # 6 task variants
  meta_roundabout/       # 6 task variants
  meta_parking/          # 6 task variants
  meta_racetrack/        # 6 task variants
```

### Generating Datasets

```bash
# Generate all environments (50k transitions each)
python benchmarks/generate_all_datasets.py

# Generate specific environment
python benchmarks/generate_all_datasets.py --env parking

# Custom transition count
python benchmarks/generate_all_datasets.py --n-transitions 100000

# Verify existing datasets
python benchmarks/generate_all_datasets.py --verify-only
```

### Task Variants Per Environment

| Environment | Tasks | Parameter Variations |
|---|---|---|
| highway | easy, dense, fast, dangerous, slow_dense, mixed | density, speed range, collision reward |
| intersection | default, dense, sparse, fast, dangerous, slow | num_vehicles, speed, collision reward |
| merge | default, dense, sparse, fast, dangerous, calm | density, merge vehicles, speed |
| roundabout | default, dense, sparse, fast, dangerous, slow | num_vehicles, speed range |
| parking | near, far, angled, reverse, center, tight | goal position, lot size |
| racetrack | default, wide, narrow, fast, slow, crowded | road width, speed range, num_vehicles |

### Loading Datasets

```python
import numpy as np

data = np.load("benchmarks/datasets/meta_highway/easy_random_seed0.npz")

observations = data["observations"]        # (50000, 25)
actions = data["actions"]                  # (50000, 2)
rewards = data["rewards"]                  # (50000,)
next_observations = data["next_observations"]  # (50000, 25)
terminals = data["terminals"]              # (50000,)  -- crashed/success
truncated = data["truncated"]              # (50000,)  -- max_steps hit

# Episode boundaries
episode_ends = np.where((terminals > 0.5) | (truncated > 0.5))[0]
```

## Rendering

The JAX environments have **no rendering** — pygame is intentionally absent so
that `jit`/`vmap` rollouts stay pure-functional.  See the top-level
`README.md` "Rendering — Known Issue" section for the current status of
the rendering pipeline.

## Running Tests

```bash
python tests/test_jax_envs_rollout.py
# Plus the full causal regression
python /tmp/test_metahighway_full.py
```

Runs 30 tests (5 per environment) verifying reset, step, auto-reset, rollout, and action effects.
