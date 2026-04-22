<h1 align="center">MetaHighwayEnv</h1>

<p align="center">
  <img src="MetaHighway.png" alt="MetaHighwayEnv — causal meta-RL for autonomous driving" width="850"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"/></a>
  <a href="https://github.com/google/jax"><img src="https://img.shields.io/badge/accelerator-JAX-informational.svg" alt="JAX"/></a>
  <a href="https://github.com/Farama-Foundation/Gymnasium"><img src="https://img.shields.io/badge/API-Gymnasium-brightgreen.svg" alt="Gymnasium"/></a>
  <a href="https://github.com/eleurent/highway-env"><img src="https://img.shields.io/badge/based%20on-highway--env-lightgrey.svg" alt="Based on highway-env"/></a>
</p>

<p align="center">
  <em>
    A causal meta-reinforcement learning environment built on
    <a href="https://github.com/eleurent/highway-env">highway-env</a>, designed for
    research on context-based adaptation and causal inference in non-stationary
    traffic scenarios. It ships with a pure-JAX reimplementation of the core
    simulator for GPU-accelerated training and a declarative
    structural-causal-model (SCM) engine with counterfactual replay and
    instrumental-variable support.
  </em>
</p>

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Reproducible environment](#reproducible-environment)
- [Quickstart](#quickstart)
- [How to create environments](#how-to-create-environments)
- [Quick reference](#quick-reference)
- [Basic usage](#basic-usage)
- [JAX-accelerated environment](#jax-accelerated-environment)
- [Causal features](#causal-features)
- [Meta-RL protocol](#meta-rl-protocol)
- [Design notes](#design-notes)
- [Known Limitations](#known-limitations)
- [Rendering — Known Issue](#rendering--known-issue)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Overview

MetaHighwayEnv extends highway-env with three layers of infrastructure for
causal meta-RL experiments:

1. **Meta-RL task system** -- task distributions, sampling strategies
   (uniform / curriculum / adaptive), context collection, and task splits
   for few-shot evaluation.

2. **Causal intervention API** -- a unified `do(X := x)` interface across
   **all 14** environment types (5 core + 9 variants) via
   `CausalInterventionMixin`, with per-env interventionable variables
   (mid-episode and episode-level), reward decomposition, and intervention
   logging.

3. **Structural Causal Model (SCM)** -- a latent-confounder weather model
   that induces confounding between speed limit, driver aggressiveness, and
   traffic density, with an instrumental variable for identification, a
   declarative SCM graph engine, and a counterfactual replay engine.

### Feature summary

| Category | Feature | Module |
|----------|---------|--------|
| Meta-RL | Task distribution (param variation) | `task_distribution.py` |
| Meta-RL | Uniform / curriculum / adaptive sampling | `task_sampler.py` |
| Meta-RL | Context episode collection | `meta_env.py` |
| Meta-RL | Task sampling (`sample_task`, `sample_tasks`, `sample_fresh_task`) | `task_distribution.py` |
| Meta-RL | Per-task reward weight customisation | `task_distribution.py` |
| Meta-RL | Few-shot context/query episode structure | `meta_env.py` |
| Causal | `CausalInterventionMixin` -- unified API for all envs | `causal_mixin.py` |
| Causal | `intervene(variable, value)` -- do-calculus | all env files |
| Causal | Mid-episode + episode-level granularity | all env files |
| Causal | Reward decomposition (`info["reward_components"]`) | all env files |
| Causal | Latent confounder (weather SCM) | `task_distribution.py` |
| Causal | Instrumental variable (road construction) | `scm.py` |
| Causal | Declarative SCM graph engine | `scm.py` |
| Causal | Nonlinear SCM (sigmoid/quadratic/threshold) | `scm.py` |
| Causal | Counterfactual replay engine | `counterfactual.py` |
| Causal | Within-episode + task-level DAG specs | `causal_graph.py` |
| Causal | Persistent intervention history | `meta_env.py` |
| Perf | Headless fast-rollout mode (`_headless` flag) | `abstract.py` |
| Perf | `enable_causal_logging` toggle | `meta_env.py` |
| Perf | **JAX-native environment** (GPU-accelerated, 25-211x faster) | `highway_env/jax/` |
| Perf | `jax.vmap` multi-task parallelism (near-free scaling) | `highway_env/jax/meta_env.py` |
| Perf | `jax.lax.scan` rollout collection | `highway_env/jax/meta_env.py` |

---

## Installation

```bash
git clone https://github.com/brunoleej/MetaHighwayEnv.git
cd MetaHighwayEnv
pip install -e .
```

Core dependencies (pulled in automatically via `pyproject.toml`):
`gymnasium`, `numpy`, `pygame`, `matplotlib`, `pandas`, `scipy`.

**For the JAX-accelerated environment** (recommended for training on
GPU), install the optional `jax` extra:

```bash
pip install -e '.[jax]'
```

This pulls in `jax`, `jaxlib`, `flax`, `optax`, and `chex`.

> JAX GPU support requires matching CUDA drivers. See
> [JAX installation](https://github.com/google/jax#installation) for
> platform-specific wheels.

**For running the benchmark algorithms** (`benchmarks/*.py`) you may
additionally want PyTorch and a handful of utilities:

```bash
pip install -e '.[meta_rl]'  # torch, tensorboard, tqdm, scikit-learn
```

---

## Reproducible environment

Use the provided `environment.yml` to create an exact conda environment:

```bash
conda env create -f environment.yml
conda activate meta-highway-env
pip install -e .
```

This creates a `meta-highway-env` conda environment with all core dependencies pinned. Pinned to verified stable combinations. Optional JAX and benchmark dependencies are listed (commented out) in `environment.yml`.

---

## Quickstart

After installation (`pip install -e .`), run the bundled example:

```bash
python examples/basic_highway.py
```

Or paste directly into a Python session:

```python
from highway_env.meta_rl.meta_env import MetaHighwayEnv

env = MetaHighwayEnv(env_type="highway", n_tasks=10, seed=42)
obs, info = env.reset()

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

For a causal intervention:

```python
env = MetaHighwayEnv(env_type="highway", n_tasks=10,
                     enable_causal_logging=True, seed=0)
env.reset()
env.intervene("driver_aggressiveness", 0.9)
print(env.get_intervention_log())
env.close()
```

See `examples/basic_highway.py` for the full annotated example covering
meta-RL rollouts, causal interventions, and normalized score computation.

---

## How to create environments

`MetaHighwayEnv` is the primary entry-point for meta-RL and causal
experiments.  You construct it directly (it is **not** a Gymnasium
registered ID):

> For single-task RL without meta-learning, use the upstream
> [highway-env](https://github.com/eleurent/highway-env) package
> directly. With **gymnasium ≥ 1.0**, you must import the package
> once before calling `gymnasium.make`:
>
> ```python
> import highway_env  # registers all envs (highway-v0, merge-v0, etc.)
> import gymnasium as gym
> env = gym.make("highway-v0")
> ```
>
> The `[project.entry-points."gymnasium.envs"]` mechanism declared in
> `pyproject.toml` is honored only by gymnasium ≤ 0.29; later versions
> dropped automatic plugin discovery.

```python
from highway_env.meta_rl import MetaHighwayEnv

env = MetaHighwayEnv(
    env_type="highway",     # see "Supported environment types" table below
    n_tasks=10,
    context_length=5,
    seed=42,
)

task = env.sample_task()
env.set_task(task)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

> **Note:** `MetaHighwayEnv` inherits from `gymnasium.Env` (via
> `import gymnasium as gym`) and follows the standard gymnasium API:
> `reset()` returns `(obs, info)` and `step()` returns the 5-tuple
> `(obs, reward, terminated, truncated, info)`.  Use `done = terminated or truncated`
> if you need the old-gym-style boolean.

**`MetaHighwayEnv` constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_type` | str | `"highway"` | Base environment type (see table below) |
| `task_distribution` | TaskDistribution | None | Task distribution (auto-creates `HighwayTaskDistribution` if None) |
| `task_sampler` | TaskSampler | None | Sampling strategy (auto-creates `UniformTaskSampler` if None) |
| `n_tasks` | int | 10 | Number of tasks (used only when `task_distribution` is None) |
| `context_length` | int | 100 | Episodes before automatic task switch |
| `include_task_id` | bool | False | Append task ID to observation vector |
| `enable_causal_logging` | bool | False | Expose `info["task"]` (confounder data) and record intervention history |
| `seed` | int | None | Random seed |

> **Important:** Set `enable_causal_logging=True` when using causal
> features (confounder analysis, intervention history).  It is off by
> default to avoid per-step overhead during headless training.

---

## Quick reference

### Observation and action spaces

| Property | Value |
|----------|-------|
| Observation type | `Kinematics` (default) — 5×5 float32 matrix (ego + 4 nearest vehicles, 5 features each) |
| Action type | `DiscreteMetaAction` — `Discrete(5)`: LANE_LEFT=0, IDLE=1, LANE_RIGHT=2, FASTER=3, SLOWER=4 |
| Reward range | [0, 1] when `normalize_reward=True` (default) |
| Termination | Ego vehicle crashed (or off-road if `offroad_terminal=True`) |
| Truncation | Episode time reaches `duration` (default 40 s) |

### Supported environment types

`MetaHighwayEnv` supports these `env_type` values.  **All environments**
include the full causal intervention API via `CausalInterventionMixin`:

**Core types:**

| `env_type` | Class | Mid-episode vars | Episode-level vars |
|------------|-------|------------------|-------------------|
| `"highway"` | `HighwayEnv` | 8 | 3 |
| `"intersection"` | `IntersectionEnv` | 8 | 2 |
| `"merge"` | `MergeEnv` | 8 | 1 |
| `"roundabout"` | `RoundaboutEnv` | 7 | 1 |
| `"racetrack"` | `RacetrackEnv` | 7 | 3 |

**Variant / extended types:**

| `env_type` | Class | Inherits from | Notes |
|------------|-------|---------------|-------|
| `"exit"` | `ExitEnv` | `HighwayEnv` | Highway exit manoeuvre |
| `"highway-fast"` | `HighwayEnvFast` | `HighwayEnv` | Lower sim freq, fewer vehicles |
| `"intersection-continuous"` | `ContinuousIntersectionEnv` | `IntersectionEnv` | Continuous action space |
| `"intersection-multi-agent"` | `MultiAgentIntersectionEnv` | `IntersectionEnv` | Multi-agent |
| `"lane-keeping"` | `LaneKeepingEnv` | `AbstractEnv` | Continuous steering, no traffic |
| `"racetrack-large"` | `RacetrackEnvLarge` | `RacetrackEnv` | Larger map, 3 lanes |
| `"racetrack-oval"` | `RacetrackEnvOval` | `RacetrackEnv` | Oval track with roadblocks |
| `"two-way"` | `TwoWayEnv` | `AbstractEnv` | Two-way traffic, overtaking |
| `"u-turn"` | `UTurnEnv` | `AbstractEnv` | U-turn manoeuvre |

Variant types inherit the full causal API from their parent class.
`LaneKeepingEnv` has its own `CausalInterventionMixin` with
environment-specific variables.

Every environment exposes `intervene()`, `intervene_batch()`,
`interventionable_variables()`, `get_causal_state()`, and
`decompose_reward()`.

---

## Basic usage

### 1. Create a meta-RL environment and sample tasks

```python
from highway_env.meta_rl import (
    MetaHighwayEnv,
    HighwayTaskDistribution,
    UniformTaskSampler,
)

# Build a task distribution that varies traffic density and reward weights
task_dist = HighwayTaskDistribution(
    n_tasks=20,
    vary_params=["vehicle_density", "collision_reward", "high_speed_reward"],
    seed=42,
)

# Wrap with meta-RL env (default context_length=100)
env = MetaHighwayEnv(
    env_type="highway",
    task_distribution=task_dist,
    context_length=5,       # context episodes before auto task switch
    seed=42,
)

# Sample and set a task
task = env.sample_task()
env.set_task(task)
obs = env.reset()       # returns np.ndarray (not a tuple)
print(f"Task {task.task_id}: difficulty={task.difficulty:.2f}")
print(f"  params = {task.params}")
```

### 2. Step through an episode and read reward components

```python
for step in range(50):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)  # always 4-tuple

    # Reward decomposition: which component drove the reward?
    rc = info.get("reward_components", {})
    for name, comp in rc.items():
        print(f"  {name}: raw={comp['raw']:.3f}  weight={comp['weight']:.2f}"
              f"  weighted={comp['weighted']:.3f}")

    if done:
        break
```

**Reward components** returned in `info["reward_components"]`:

| Component | Description |
|-----------|-------------|
| `collision_reward` | 1.0 if crashed, else 0.0 (weight default: -1) |
| `high_speed_reward` | Scaled forward speed ∈ [0, 1] (weight default: 0.4) |
| `right_lane_reward` | Normalised lane index (weight default: 0.1) |
| `lane_change_reward` | 1.0 if lane change action, else 0.0 (weight default: 0) |
| `on_road_reward` | 1.0 if on road; multiplies total reward |

### 3. Train/test split for meta-learning evaluation

```python
train_tasks = task_dist.sample_tasks(n=10)
test_tasks  = task_dist.sample_tasks(n=5)

# Freeze task during evaluation (no auto-switching)
env.enable_meta_testing()
env.set_task(test_tasks[0])
obs = env.reset()
# ... evaluate adaptation performance ...
env.disable_meta_testing()
```

> **Note:** `sample_tasks(n)` is available on `HighwayTaskDistribution`.
> `CausalTaskDistribution` provides `sample_task()` (singular) and
> `sample_fresh_task(task_id)` for generating unseen tasks.

---

## JAX-accelerated environment

MetaHighwayEnv includes a pure-JAX reimplementation of the highway
environment for GPU-accelerated training. This eliminates the CPU-bound
simulation bottleneck, providing **25-211x speedup** over the standard
Gymnasium environment.

### Architecture

The JAX environment follows the [gymnax](https://github.com/RobertTLange/gymnax)
pattern: all functions are pure-functional, static, and compatible with
`jax.jit`, `jax.vmap`, and `jax.lax.scan`.

| Component | Module | Description |
|-----------|--------|-------------|
| Data structures | `state.py` | `EnvState`, `EnvParams`, `VehicleState` + per-env params/state (`flax.struct.dataclass`) |
| Kinematics | `kinematics.py` | Kinematic bicycle model (vmapped) |
| Traffic | `traffic.py` | IDM acceleration + simplified MOBIL lane change |
| Collision | `collision.py` | AABB (Axis-Aligned Bounding Box) detection |
| Observation | `observation.py` | Pandas-free kinematic observation |
| Reward | `reward.py` | Highway reward function |
| Environment | `env.py` | `HighwayJaxEnv` (reset/step/step_auto_reset) |
| Environment | `env_merge.py` | `MergeJaxEnv` — highway + merge lane |
| Environment | `env_intersection.py` | `IntersectionJaxEnv` — 4-way intersection |
| Environment | `env_roundabout.py` | `RoundaboutJaxEnv` — circular road |
| Environment | `env_racetrack.py` | `RacetrackJaxEnv` — oval track with occupancy grid |
| Meta-env | `meta_env.py` | Task sampling, `collect_rollout`, vmapped batched collection |

> **Note:** `collect_rollout_batched` currently supports `HighwayJaxEnv` only.

### Quick start

```python
from highway_env.jax import HighwayJaxEnv, EnvParams, N_OBS
import jax
import jax.numpy as jnp
import jax.random as jrandom

params = EnvParams()  # default highway parameters
key = jrandom.PRNGKey(42)

# Reset
obs, state = HighwayJaxEnv.reset(key, params)
# obs shape: (25,) = N_OBS(5) * 5 features [presence, x, y, vx, vy]

# Step
action = jnp.array([1.0, 0.0])  # [acceleration, steering]
key, subkey = jrandom.split(key)
obs, state, reward, done, info = HighwayJaxEnv.step(subkey, state, action, params)

# Auto-reset step (for continuous rollouts)
obs, state, reward, done, info = HighwayJaxEnv.step_auto_reset(subkey, state, action, params)
```

### Vectorized multi-task collection

```python
from highway_env.jax import (
    generate_task_batch, collect_rollout, collect_rollout_batched
)

# Generate 10 random task parameterizations
key = jrandom.PRNGKey(0)
batch_params = generate_task_batch(key, EnvParams(), n_tasks=10)

# Define a JAX-compatible policy
def random_policy(key, obs):
    return jrandom.uniform(key, (2,), minval=-1.0, maxval=1.0)

# Collect 200 steps across all 10 tasks in parallel (via jax.vmap)
transitions = collect_rollout_batched(key, batch_params, random_policy, 200, 10)
# transitions["obs"].shape == (10, 200, 25)
# transitions["rewards"].shape == (10, 200)
```

### Per-environment observation and action spaces

| `env_type` | Observation | obs_dim | action_dim | JAX env class | Notes |
|------------|-------------|---------|------------|---------------|-------|
| `highway` | Kinematics (5×5) | 25 | 2 | `HighwayJaxEnv` | Default |
| `intersection` | Kinematics (15×7) | 105 | 2 | `IntersectionJaxEnv` | Larger observation |
| `merge` | Kinematics (5×5) | 25 | 2 | `MergeJaxEnv` | Highway + merge lane |
| `roundabout` | Kinematics (5×5) | 25 | 2 | `RoundaboutJaxEnv` | Circular road |
| `racetrack` | OccupancyGrid (2×12×12) | 288 | 2 | `RacetrackJaxEnv` | Steering only (accel ignored) |

> **Note:** All 5 Meta-Highway environments support `--jax-env` for
> GPU-accelerated training. The JAX backend auto-detects obs_dim and
> action_dim from the environment.

> **JAX action spaces:** The JAX backend uses `DiscreteMetaAction` (5 actions:
> LANE_LEFT=0, IDLE=1, LANE_RIGHT=2, FASTER=3, SLOWER=4) by default for
> `highway`, `merge`, `roundabout`, and `intersection` (3 actions: SLOWER=0,
> IDLE=1, FASTER=2). Use `step_discrete(action_int)` for discrete envs.
> Only `racetrack` uses a continuous 2D action `[accel, steering]`.

### Performance

Benchmarks on a single GPU (after JIT warmup):

| Scenario | Gymnasium | JAX | Speedup |
|----------|-----------|-----|---------|
| Single task (200 steps) | 106.3s | 4.2s | **25x** |
| 10-task vmap (2000 total steps) | ~1063s | 5.0s | **211x** |
| PEARL training (2 iterations, fast-debug) | ~465s | 17.5s | **~27x** |

The `jax.vmap` parallelism is nearly free: 10 tasks take only ~20% longer
than 1 task, giving **211x throughput** vs sequential Gymnasium execution.

### Design decisions

- **Fixed-size arrays** (`N_MAX=50`): JAX JIT requires static shapes. An
  `active` mask handles dynamic vehicle counts.
- **Compile-time constants**: `N_OBS=5` (observed vehicles), `N_SIM_PER_POLICY=15`
  (simulation sub-steps per policy step) are module-level constants, not
  `EnvParams` fields, because `jax.lax.scan` needs static loop lengths.
- **AABB collision**: Simpler and more JIT-friendly than SAT polygon
  intersection. Sufficient for axis-aligned highway vehicles.
- **Simplified MOBIL**: Core lane-change logic preserved; full MOBIL
  politeness model simplified for JIT compatibility.

---

## Causal features

### 4. do-Calculus intervention API

```python
# Interventions can be called on env directly (delegates to base_env)
# or on env.base_env directly

# Mid-episode: change driver aggressiveness (takes effect immediately)
meta = env.base_env.intervene("driver_aggressiveness", 0.9)
# meta == {"variable": "driver_aggressiveness", "value": 0.9,
#          "granularity": "mid_episode", "applied": True, ...}

# Mid-episode: lower speed limit on all lanes
env.base_env.intervene("speed_limit", 20.0)

# Episode-level: change lane count (needs reset)
meta = env.base_env.intervene("lanes_count", 2)
assert meta["requires_reset"]
obs = env.reset()   # scene rebuilt with 2 lanes

# Batch intervention (counterfactual task)
results = env.base_env.intervene_batch({
    "driver_aggressiveness": 0.1,
    "collision_reward": -0.5,
    "lanes_count": 5,
})
if any(r["requires_reset"] for r in results.values()):
    obs = env.reset()

# Introspection: which variables can be intervened on?
print(env.base_env.interventionable_variables())
# {"collision_reward": "mid_episode", "driver_aggressiveness": "mid_episode",
#  "lanes_count": "episode_level", ...}
```

**Interventionable variables per environment:**

All environments share `driver_aggressiveness` and `speed_limit` as
common mid-episode variables (handled generically by `CausalInterventionMixin`).
Environment-specific variables are listed below.

**HighwayEnv** (11 total):

| Variable | Granularity | Effect |
|----------|-------------|--------|
| `driver_aggressiveness` | mid-episode | Mutates IDM/MOBIL params on all traffic vehicles |
| `speed_limit` | mid-episode | Sets `lane.speed_limit` on every lane |
| `collision_reward` | mid-episode | Updates reward weight |
| `high_speed_reward` | mid-episode | Updates reward weight |
| `right_lane_reward` | mid-episode | Updates reward weight |
| `lane_change_reward` | mid-episode | Updates reward weight |
| `reward_speed_range` | mid-episode | Updates `[low, high]` speed mapping |
| `duration` | mid-episode | Changes episode time limit |
| `vehicles_count` | episode-level | Requires `reset()` |
| `vehicles_density` | episode-level | Requires `reset()` |
| `lanes_count` | episode-level | Requires `reset()` |

**IntersectionEnv** (10 total):

| Variable | Granularity |
|----------|-------------|
| `driver_aggressiveness`, `speed_limit`, `collision_reward`, `high_speed_reward`, `arrived_reward`, `reward_speed_range`, `duration`, `spawn_probability` | mid-episode |
| `initial_vehicle_count`, `destination` | episode-level |

**MergeEnv** (9 total):

| Variable | Granularity |
|----------|-------------|
| `driver_aggressiveness`, `speed_limit`, `collision_reward`, `high_speed_reward`, `right_lane_reward`, `lane_change_reward`, `merging_speed_reward`, `reward_speed_range` | mid-episode |
| `other_vehicles_type` | episode-level |

**RoundaboutEnv** (8 total):

| Variable | Granularity |
|----------|-------------|
| `driver_aggressiveness`, `speed_limit`, `collision_reward`, `high_speed_reward`, `right_lane_reward`, `lane_change_reward`, `duration` | mid-episode |
| `incoming_vehicle_destination` | episode-level |

**RacetrackEnv** (10 total):

| Variable | Granularity | Effect |
|----------|-------------|--------|
| `driver_aggressiveness` | mid-episode | Mutates IDM/MOBIL params on all traffic vehicles |
| `speed_limit` | mid-episode | Sets `lane.speed_limit` on every lane |
| `collision_reward` | mid-episode | Updates reward weight |
| `lane_centering_cost` | mid-episode | Updates lane centering penalty weight |
| `lane_centering_reward` | mid-episode | Updates lane centering reward |
| `action_reward` | mid-episode | Updates action penalty weight |
| `duration` | mid-episode | Changes episode time limit |
| `other_vehicles` | episode-level | Requires `reset()` |
| `controlled_vehicles` | episode-level | Requires `reset()` |
| `terminate_off_road` | episode-level | Requires `reset()` |

**LaneKeepingEnv** (4 total):

| Variable | Granularity | Effect |
|----------|-------------|--------|
| `state_noise` | mid-episode | Observation noise magnitude |
| `derivative_noise` | mid-episode | Derivative noise magnitude |
| `simulation_frequency` | episode-level | Requires `reset()` |
| `policy_frequency` | episode-level | Requires `reset()` |

> **Note:** `LaneKeepingEnv` uses continuous steering control with a
> BicycleVehicle and has no traffic vehicles.
> `driver_aggressiveness` and `speed_limit` have no effect.

**ExitEnv, HighwayEnvFast** inherit all variables from `HighwayEnv`.
**ContinuousIntersectionEnv, MultiAgentIntersectionEnv** inherit from `IntersectionEnv`.
**RacetrackEnvLarge, RacetrackEnvOval** inherit from `RacetrackEnv`.

### 5. Latent confounder and reading confounder values

```python
from highway_env.meta_rl import CausalTaskDistribution, MetaHighwayEnv

# Create a causal task distribution with latent confounder
causal_dist = CausalTaskDistribution(
    n_tasks=20,
    noise_scale=0.05,
    confounder_distribution="uniform",  # or "discrete" for {clear, rain, fog}
    extra_params={"lanes_count": 4},
    seed=42,
)

# enable_causal_logging=True is REQUIRED to populate info["task"]
env = MetaHighwayEnv(
    env_type="highway",
    task_distribution=causal_dist,
    enable_causal_logging=True,
)

task = causal_dist.tasks[0]
env.set_task(task)
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())

# Agent observation does NOT contain the confounder.
# But info["task"]["confounder"] exposes it for analysis:
print(info["task"])
# {"task_id": 0, "difficulty": 0.18,
#  "confounder": {"confounder_weather": 0.18}}
```

**SCM (Structural Causal Model) — Linear (default):**

```
weather (latent) in [0, 1]
  |-- speed_limit           = 30 - 15*weather + noise     in [15, 30]
  |-- driver_aggressiveness = 0.1 + 0.8*weather + noise   in [0, 1]
  +-- vehicles_density      = 2.0 - 1.5*weather + noise   in [0.5, 2.0]
```

**SCM — Nonlinear (recommended for paper experiments):**

```
weather (latent) in [0, 1]
  |-- speed_limit           = sigmoid(weather, w=-15, b=30)  + noise   in [15, 30]
  |-- driver_aggressiveness = threshold(weather, w=0.8, b=0.1) + noise in [0, 1]
  +-- vehicles_density      = quadratic(weather, w=-1.5, b=2.0) + noise in [0.5, 2.0]
```

Use `default_highway_scm_nonlinear()` for the nonlinear version (see §6
below).

### 6. Declarative SCM graph engine with instrumental variable

```python
from highway_env.meta_rl import SCMGraph, CausalTaskDistribution

# Default SCM with instrumental variable
scm = SCMGraph.default_highway_scm_with_iv(seed=42)
print(scm.describe())
# SCM Graph:
#   weather [latent] ~ uniform[0.0, 1.0]  (root)
#   road_construction [observed] ~ uniform[0.0, 1.0]  (root)
#   speed_limit [observed] = 30.0-15.0*weather + noise
#   driver_aggressiveness [observed] = 0.1+0.8*weather + noise
#   vehicles_density [observed] = 2.0-1.5*weather + 0.0-0.3*road_construction + noise

# Observational sample
sample = scm.sample()

# Interventional sample: do(speed_limit := 25)
sample_do = scm.sample(interventions={"speed_limit": 25.0})

# Batch sample: generate n realisations at once
samples = scm.sample_batch(n=100)

# Export Graphviz DOT for paper figures
dot_str = scm.to_dot()

# Build a custom SCM
custom = SCMGraph(noise_scale=0.1, seed=0)
custom.add_node("weather", "latent", 0.0, 1.0)
custom.add_node("road_quality", "latent", 0.0, 1.0)
custom.add_node("speed_limit", "observed", 15.0, 30.0)
custom.add_node("aggressiveness", "observed", 0.0, 1.0)
custom.add_edge("weather", "speed_limit", weight=-15.0, intercept=30.0)
custom.add_edge("weather", "aggressiveness", weight=0.8, intercept=0.1)
custom.add_edge("road_quality", "aggressiveness", weight=-0.3, intercept=0.0)

# Generate tasks from any SCM
dist = CausalTaskDistribution.from_scm_graph(scm, n_tasks=50)
```

### 6b. Linear vs. nonlinear SCM selection

The SCM engine provides **both** linear and nonlinear factories.  For
paper experiments we recommend the **nonlinear** version, which uses
sigmoid, threshold, and quadratic mechanisms to produce clearly
nonlinear causal relationships.  The linear version is useful as an
ablation baseline.

```python
from highway_env.meta_rl import SCMGraph, CausalTaskDistribution

# ── Linear SCM (ablation / baseline) ──
scm_lin = SCMGraph.default_highway_scm(seed=42)

# ── Nonlinear SCM (recommended for main results) ──
scm_nl = SCMGraph.default_highway_scm_nonlinear(seed=42)

# With instrumental variable:
scm_nl_iv = SCMGraph.default_highway_scm_nonlinear_with_iv(seed=42)

print(scm_nl.describe())
# SCM Graph:
#   weather [latent] ~ uniform[0.0, 1.0]  (root)
#   speed_limit [observed] = sigmoid(weather, w=-15.0, b=30.0) + ε  ...
#   driver_aggressiveness [observed] = threshold(weather, w=0.8, b=0.1) + ε  ...
#   vehicles_density [observed] = quadratic(weather, w=-1.5, b=2.0) + ε  ...

# Generate tasks from nonlinear SCM
dist = CausalTaskDistribution.from_scm_graph(scm_nl_iv, n_tasks=100)
```

**Available mechanisms:**

| Mechanism | Function | Formula | Use case |
|-----------|----------|---------|----------|
| `linear` | `_linear_mechanism` | `b + w*x` | Default / ablation |
| `sigmoid` | `_sigmoid_mechanism` | `b + w*σ(10*(x−0.5))` | Saturating / S-curve |
| `quadratic` | `_quadratic_mechanism` | `b + w*x²` | Accelerating / convex |
| `threshold` | `_threshold_mechanism` | `b + w*1[x≥0.5]` | Regime switching |

Custom mechanisms can be passed to any edge via `mechanism=my_fn`.

### 7. Counterfactual replay

```python
from highway_env.meta_rl import CounterfactualEngine

engine = CounterfactualEngine()
obs = env.reset()

# Run a factual trajectory and checkpoint at step 5
for t in range(10):
    obs, r, done, info = env.step(policy(obs))
    if t == 5:
        checkpoint = engine.save_state(env)
    if done:
        break

# Compare factual vs. counterfactual (do(aggressiveness := 0.1))
result = engine.compare_factual_counterfactual(
    env, checkpoint,
    factual_policy=policy,
    counterfactual_interventions={"driver_aggressiveness": 0.1},
    steps=15,
)
print(f"Factual return:       {result['factual']['total_return']:.3f}")
print(f"Counterfactual return: {result['counterfactual']['total_return']:.3f}")
print(f"Causal effect (delta): {result['delta_return']:.3f}")

# The environment is automatically restored to its pre-rollout state
```

### 8. Intervention history for cross-episode causal analysis

```python
# Create env with causal logging enabled
env = MetaHighwayEnv(
    env_type="highway",
    task_distribution=causal_dist,
    enable_causal_logging=True,    # required for intervention logging
)

# interventions are logged persistently across episodes
env.intervene("driver_aggressiveness", 0.9)
env.intervene("speed_limit", 20.0)
obs = env.reset()  # new episode
env.intervene("collision_reward", -2.0)

history = env.get_full_intervention_history()
# [{"episode": 1, "task_id": 0, "variable": "driver_aggressiveness", ...},
#  {"episode": 1, "task_id": 0, "variable": "speed_limit", ...},
#  {"episode": 2, "task_id": 0, "variable": "collision_reward", ...}]
```

### 9. Causal DAG export

The within-episode and task-level causal DAGs are specified declaratively
in `highway_env/meta_rl/causal_graph.py` and can be exported to Graphviz
DOT format:

```python
from highway_env.meta_rl import WITHIN_EPISODE_DAG, TASK_LEVEL_DAG, dag_to_dot, get_full_two_level_dag

# Export step-level DAG to DOT for paper figures
dot_str = dag_to_dot(WITHIN_EPISODE_DAG, name="StepDAG")
with open("step_dag.dot", "w") as f:
    f.write(dot_str)
# Then: dot -Tpdf step_dag.dot -o step_dag.pdf

# Full two-level DAG (task + step combined)
full = get_full_two_level_dag()
```

---

## Meta-RL protocol

A standard meta-RL experiment loop with MetaHighwayEnv:

```python
from highway_env.meta_rl import (
    MetaHighwayEnv,
    CausalTaskDistribution,
    CurriculumTaskSampler,
)

# 1. Define causal task distribution
task_dist = CausalTaskDistribution(n_tasks=100, seed=42)

# 2. Create meta-RL env with curriculum sampling
sampler = CurriculumTaskSampler(task_dist, seed=42)
env = MetaHighwayEnv(
    env_type="highway",
    task_distribution=task_dist,
    task_sampler=sampler,
    context_length=5,
)

# 3. Meta-training loop
for meta_iter in range(1000):
    task = env.sample_task()
    env.set_task(task)

    # Collect context (adaptation) episodes
    context = []
    for ep in range(5):
        obs = env.reset()
        episode_data = []
        done = False
        while not done:
            action = explore_policy(obs)
            obs, reward, done, info = env.step(action)
            episode_data.append((obs, action, reward, info))
        context.append(episode_data)

    # Adapted policy execution
    env.enable_meta_testing()
    obs = env.reset()
    done = False
    while not done:
        action = adapted_policy(obs, context)
        obs, reward, done, info = env.step(action)
    env.disable_meta_testing()

# 4. Meta-evaluation on held-out tasks
eval_tasks = [task_dist.sample_fresh_task(i) for i in range(20)]
for task in eval_tasks:
    env.set_task(task)
    env.enable_meta_testing()
    # ... evaluate adaptation ...
```

### Additional MetaHighwayEnv methods

| Method | Description |
|--------|-------------|
| `env.get_context(n_episodes=None)` | Retrieve collected context episode data |
| `env.get_task_performance(task_id=None)` | Average returns per task |
| `env.get_causal_state()` | Structured dict of all causal variables + ego/traffic state |
| `env.decompose_reward(info)` | Decomposed reward components from last step |
| `env.interventionable_variables()` | Variable → granularity mapping (env-type aware) |
| `env.get_intervention_log()` | Session intervention log (requires `enable_causal_logging`) |
| `env.clear_intervention_log()` | Clear session log |
| `env.get_full_intervention_history()` | Persistent history across episodes |
| `env.clear_full_intervention_history()` | Clear persistent history |

---

## Design notes

### Headless fast-rollout mode

When `render_mode` is `None` (default for training), a `_headless` flag
in `AbstractEnv` skips the inner-loop `_automatic_rendering()` call in
`_simulate()`, eliminating per-sub-step rendering overhead (typically 15
sub-steps per policy step at `simulation_frequency=15`).

Additionally, `enable_causal_logging=False` (default) skips per-step
`info["task"]` construction and intervention log writes, further reducing
overhead during headless training.

### Two-level causal structure

MetaHighwayEnv has a two-level causal structure that C2A must disentangle:

```
Task level (between episodes):
  weather (latent) --> speed_limit
                   --> driver_aggressiveness
                   --> vehicles_density  <-- road_construction (IV)

Step level (within episode):
  action --> ego_speed --> ego_position --> traffic_state --> collision
                                       |                  |
  driver_aggressiveness ---------------+                  +--> reward
  speed_limit ---------+                                  |
                       +-------> speed_reward ------------+
```

The formal DAG specifications are in `highway_env/meta_rl/causal_graph.py`
and can be exported to Graphviz DOT format via `dag_to_dot()`.

### Confounder hiding mechanism

Task parameters with an underscore prefix (`_confounder_weather`) are
**never** injected into the base environment config.  The filtering happens
in `MetaHighwayEnv._configure_env_for_task()`:

```python
for param, value in task.params.items():
    if param.startswith("_"):
        continue  # hidden from agent
    config[param] = value
```

The confounder is exposed only through `info["task"]["confounder"]` for
post-hoc analysis -- it never enters the agent's observation.

> **Note:** `info["task"]` is only populated when `enable_causal_logging=True`.

### Aggressiveness mapping

`IDMVehicle.set_aggressiveness(agg)` maps a scalar `agg` in [0, 1] to
seven IDM/MOBIL parameters:

| Parameter | agg=0 (calm) | agg=1 (aggressive) |
|-----------|-------------|-------------------|
| `TIME_WANTED` | 2.0 s | 0.5 s |
| `COMFORT_ACC_MAX` | 2.0 m/s^2 | 6.0 m/s^2 |
| `COMFORT_ACC_MIN` | -3.0 m/s^2 | -7.0 m/s^2 |
| `DISTANCE_WANTED` | 7+L m | 2+L m |
| `POLITENESS` | 0.5 | 0.0 |
| `LANE_CHANGE_MIN_ACC_GAIN` | 0.1 m/s^2 | 1.0 m/s^2 |
| `LANE_CHANGE_MAX_BRAKING_IMPOSED` | 1.5 m/s^2 | 4.0 m/s^2 |

### `gym` vs `gymnasium` API

`MetaHighwayEnv` inherits from `gymnasium.Env` (via `import gymnasium as gym`)
and exposes the standard **gymnasium 5-tuple** API end-to-end.  If you are
porting old-`gym`-style code that expects a single `done` flag, derive it
from the 5-tuple:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated  # old-gym compatibility
```

| Layer | Package | `reset()` returns | `step()` returns |
|-------|---------|-------------------|------------------|
| Base environments | `gymnasium` | `(obs, info)` | `(obs, reward, terminated, truncated, info)` |
| `MetaHighwayEnv`  | `gymnasium` | `(obs, info)` | `(obs, reward, terminated, truncated, info)` |

### Project structure

```
highway_env/                       # Core library
  __init__.py                      # Gymnasium env registration (highway-v0, etc.)
  jax/                             # JAX-native envs (GPU-accelerated, 5 env types)
    state.py                       # EnvState, EnvParams, VehicleState
    kinematics.py                  # Kinematic bicycle model (vmapped)
    lane.py                        # Lane utilities (center_y, lane_id, steering)
    traffic.py                     # IDM acceleration + simplified MOBIL
    collision.py                   # AABB collision detection
    observation.py                 # Pandas-free kinematic observation
    reward.py                      # Highway reward function
    env.py                         # HighwayJaxEnv (reset/step/step_auto_reset)
    env_merge.py                   # MergeJaxEnv
    env_intersection.py            # IntersectionJaxEnv
    env_roundabout.py              # RoundaboutJaxEnv
    env_racetrack.py               # RacetrackJaxEnv (occupancy-grid obs)
    meta_env.py                    # Task sampling, collect_rollout, vmap batching
  meta_rl/                         # Meta-RL infrastructure
    meta_env.py                    # MetaHighwayEnv wrapper (gymnasium.Env, 4-tuple API)
    task_distribution.py           # Task, HighwayTaskDistribution, CausalTaskDistribution
    task_sampler.py                # Uniform / Curriculum / Adaptive samplers
    scm.py                         # SCMGraph declarative engine
    counterfactual.py              # CounterfactualEngine
    causal_graph.py                # WITHIN_EPISODE_DAG, TASK_LEVEL_DAG
  envs/                            # Environment implementations
    common/
      abstract.py                  # AbstractEnv base class (headless flag)
      causal_mixin.py              # CausalInterventionMixin (shared causal API)
    highway_env.py                 # HighwayEnv, HighwayEnvFast
    intersection_env.py            # IntersectionEnv + variants
    merge_env.py                   # MergeEnv
    roundabout_env.py              # RoundaboutEnv
    racetrack_env.py               # RacetrackEnv + variants
    exit_env.py, lane_keeping_env.py
  vehicle/
    behavior.py                    # IDMVehicle with set_aggressiveness()
  road/
    lane.py                        # StraightLane with mutable speed_limit
    road.py                        # Road, RoadNetwork

benchmarks/                        # Reference offline/causal meta-RL algorithms
  offline_pearl_jax.py             # Offline PEARL (JAX)
  online_pearl_jax.py              # Online PEARL (JAX)
  focal_jax.py                     # FOCAL — distance-metric + BRAC
  corro_jax.py                     # CoRRO — InfoNCE contrastive encoder
  causalcomrl_jax.py               # CausalCoMRL — latent causal VAE + contrastive
  prism.py                         # PRISM — dual-space causal discovery + BRAC
  csro.py                          # CSRO — context-shift robust offline meta-RL
  borel.py                         # BOReL — Bayesian offline meta-RL
  macaw.py                         # MACAW — advantage-weighted offline meta-RL
  iql_baseline.py                  # IQL baseline
  jax_sac.py                       # JAX SAC behaviour-policy trainer
  scripts/
    offline_pearl.sh               # Offline PEARL runner
    focal.sh                       # FOCAL runner
    corro.sh                       # CoRRO runner
    causalcomrl.sh                 # CausalCoMRL runner
    prism.sh                       # PRISM runner (5 envs × 5 seeds × 4 SCMs)
    csro.sh, borel.sh, macaw.sh    # Other baseline runners
    fill_prism_tables.py           # Aggregator for paper tables
```

### CausalInterventionMixin architecture

All 14 supported environment types include `CausalInterventionMixin`,
which provides the unified causal API:

```
CausalInterventionMixin          AbstractEnv
          |                          |
          +--- intervene()           +--- config, road, vehicle
          +--- intervene_batch()     +--- step(), reset()
          +--- interventionable_variables()
          +--- get_causal_state()
          +--- decompose_reward()
          |
          v
  Core types (own mixin):
    HighwayEnv(CausalInterventionMixin, AbstractEnv)
      +-- ExitEnv(HighwayEnv)           -- inherits mixin
      +-- HighwayEnvFast(HighwayEnv)    -- inherits mixin
    IntersectionEnv(CausalInterventionMixin, AbstractEnv)
      +-- ContinuousIntersectionEnv(IntersectionEnv)
      +-- MultiAgentIntersectionEnv(IntersectionEnv)
    MergeEnv(CausalInterventionMixin, AbstractEnv)
    RoundaboutEnv(CausalInterventionMixin, AbstractEnv)
    RacetrackEnv(CausalInterventionMixin, AbstractEnv)
      +-- RacetrackEnvLarge(RacetrackEnv)
      +-- RacetrackEnvOval(RacetrackEnv)
    LaneKeepingEnv(CausalInterventionMixin, AbstractEnv)
```

Each environment subclass declares its own `_MID_EPISODE_VARS` and
`_EPISODE_LEVEL_VARS` sets.  The mixin handles common variables
(`driver_aggressiveness`, `speed_limit`) generically; environments
override `_apply_mid_episode_intervention()` only when needed for
env-specific live-object mutation.  Variant types inherit all variables
from their parent class.

---

## Known Limitations

**`pkg_resources` deprecation warning from pygame:**
On import, pygame emits a `UserWarning` about `pkg_resources` being deprecated.
This originates from `pygame/pkgdata.py` and is not caused by this codebase.
It will be resolved by a pygame upstream release. No action is needed on our end.
To suppress the warning in the meantime:

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
import highway_env
```

**Deprecation timeline:**

- **`arrival_probability` → `spawn_probability`**: A backward-compatibility shim
  in `MetaHighwayEnv._configure_env_for_task()` silently renames
  `arrival_probability` to `spawn_probability` and emits a `DeprecationWarning`.
  The shim is active in 2.x and will be **removed in 3.0.0**.
  Update any task params or configs to use `spawn_probability` directly.

- **Dead nested-parameter path** (`_configure_env_for_task` underscore-split
  fallback): Parameters containing underscores that do not match the split
  pattern are silently ignored. This dead fallback path is scheduled for
  **removal in 3.0.0**.

**`collect_rollout_batched` scope:**
`collect_rollout_batched` (in `highway_env/jax/meta_env.py`) is currently wired
for `HighwayJaxEnv` only. Users of `MergeJaxEnv`, `IntersectionJaxEnv`,
`RoundaboutJaxEnv`, or `RacetrackJaxEnv` should write their own rollout loop
using `collect_rollout` directly (see the docstring in that module).

---

## Rendering — Known Issue

**Rendering is currently under development and is NOT production-ready.**

The JAX environment provides accurate dynamics, observations, and rewards
(verified by the test suite under `tests/`). However, the rendering pipeline
that converts JAX state into videos is currently being revised — rendered
videos do not faithfully represent the agent's actual policy behavior due
to dynamics divergence between the JAX simulator and gymnasium's pygame
renderer.

**Current status:**
- Training / evaluation work correctly (numerical metrics are trustworthy).
- Causal experiments work correctly (see `highway_env/meta_rl/`).
- Video rendering is being fixed — videos may not match policy quality.
- A proper native JAX renderer is under active development.

**Workaround:** For visualization during development, use the gymnasium
env directly (e.g., `gymnasium.make("highway-v0", render_mode="rgb_array")`).
Note that this does NOT reflect the JAX SAC policy's behavior exactly
because JAX and gymnasium use different simulator internals.

**Tracking:** This is a known issue. Rendering fixes are in progress.

### render_mode usage (gymnasium 0.28+)

```python
import gymnasium as gym

env = gym.make("highway-v0", render_mode="rgb_array")  # for video recording
env = gym.make("highway-v0", render_mode="human")      # interactive window
env = gym.make("highway-v0")                           # render_mode=None (default, training)
```

Supported `render_mode` values: `"rgb_array"` | `"human"` | `None`

---

## Contributing

Install dev hooks: `pip install pre-commit && pre-commit install`

---

## Acknowledgements

- **highway-env**: built on the excellent [highway-env](https://github.com/eleurent/highway-env)
  by Edouard Leurent — this project re-uses highway-env's simulator,
  IDM/MOBIL behaviour models, and road topologies.
- **Gymnasium**: the underlying environments follow the
  [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API.
- **JAX / Flax**: the GPU-accelerated environments are written in
  [JAX](https://github.com/google/jax) and
  [Flax](https://github.com/google/flax).
- **gymnax**: the JAX environment API pattern is inspired by
  [gymnax](https://github.com/RobertTLange/gymnax).

---

## Citation

If you use MetaHighwayEnv in your research, please cite both this
repository and the upstream highway-env project it is built on. A
ready-to-use `CITATION.cff` is provided at the repository root for
GitHub's citation widget.

```bibtex
@misc{metahighwayenv2025,
  title        = {{MetaHighwayEnv}: A Causal Meta-Reinforcement Learning
                   Environment for Autonomous Driving},
  author       = {{Jisu Lee}},
  year         = {2025},
  howpublished = {\url{https://github.com/leejisue/MetaHighwayEnv}},
  note         = {Built on highway-env by Edouard Leurent.}
}

@misc{highway-env,
  author       = {Leurent, Edouard},
  title        = {An Environment for Autonomous Driving Decision-Making},
  year         = {2018},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Farama-Foundation/HighwayEnv}},
}
```

---

## License

MetaHighwayEnv is released under the [MIT License](LICENSE). The project
is derived from [highway-env](https://github.com/eleurent/highway-env)
(also MIT-licensed) by Edouard Leurent; both copyright notices are
retained in the `LICENSE` file at the repository root.
