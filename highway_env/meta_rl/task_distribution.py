"""Task distribution definitions for meta-RL experiments."""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .constants import REWARD_PARAM_RANGES, SPEED_RANGES, PENALTY_RANGES


@dataclass
class Task:
    """Represents a single task in meta-RL."""
    task_id: int
    params: Dict[str, Any]
    description: str = ""
    difficulty: float = 0.5  # 0 to 1, for curriculum learning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "params": self.params,
            "description": self.description,
            "difficulty": self.difficulty
        }


class TaskDistribution(ABC):
    """Abstract base class for task distributions."""
    
    def __init__(self, n_tasks: int, seed: Optional[int] = None):
        self.n_tasks = n_tasks
        self.rng = np.random.default_rng(seed)
        self.tasks = self._generate_tasks()
        # Q4: O(1) task lookup index built after generation
        self._task_index: Dict[int, Task] = {t.task_id: t for t in self.tasks}

    @abstractmethod
    def _generate_tasks(self) -> List[Task]:
        """Generate the set of tasks."""
        pass

    @abstractmethod
    def sample_task(self) -> Task:
        """Sample a task from the distribution."""
        pass

    def get_task(self, task_id: int) -> Task:
        """Get a specific task by ID — O(1) dict lookup."""
        task = self._task_index.get(task_id)
        if task is None:
            raise ValueError(f"Task with ID {task_id} not found")
        return task
    
    def get_all_tasks(self) -> List[Task]:
        """Return all tasks in the distribution."""
        return self.tasks.copy()


class HighwayTaskDistribution(TaskDistribution):
    """Task distribution for Highway environments with varying parameters."""
    
    def __init__(self, 
                 n_tasks: int,
                 env_type: str = "highway",
                 vary_params: Optional[List[str]] = None,
                 param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 seed: Optional[int] = None):
        
        self.env_type = env_type
        self.vary_params = vary_params or self._get_default_vary_params()
        self.param_ranges = param_ranges or self._get_default_param_ranges()
        
        super().__init__(n_tasks, seed)
    
    # Reward-weight parameter ranges — sourced from constants.py (Q9).
    _REWARD_PARAM_RANGES: Dict[str, Tuple[float, float]] = REWARD_PARAM_RANGES

    def _get_default_vary_params(self) -> List[str]:
        """Get default parameters to vary based on environment type."""
        defaults = {
            "highway": ["vehicles_density", "reward_speed_range_low", "reward_speed_range_high", "collision_reward"],
            "intersection": ["duration", "spawn_probability", "collision_reward"],
            # NOTE: other_vehicles_type variation deferred — requires categorical sampling
            "merge": ["merging_speed_reward", "collision_reward"],
            "roundabout": ["duration", "vehicles_density", "collision_reward"]
        }
        return defaults.get(self.env_type, ["collision_reward"])
    
    def _get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get default parameter ranges.

        Includes reward weight ranges so that per-task reward
        customisation works out of the box when those params are
        listed in ``vary_params``.
        """
        ranges = {
            # Highway env parameters — speed ranges from constants (Q9)
            "reward_speed_range_low":  SPEED_RANGES["reward_speed_range_low"],
            "reward_speed_range_high": SPEED_RANGES["reward_speed_range_high"],

            # Intersection env parameters
            "duration": (10, 20),
            "spawn_probability": (0.1, 0.3),

            # Merge env parameters
            "merging_speed_reward": (0.05, 0.2),

            # Highway and Roundabout env parameters
            "vehicles_density": (0.5, 3.0),

            # Common parameters
            "lanes_count": (2, 4),
            "initial_spacing": (1, 3),
            "ego_spacing": (1.0, 2.0),
        }
        # Merge reward-weight ranges from constants (Q9)
        ranges.update(self._REWARD_PARAM_RANGES)
        return ranges
    
    def _generate_tasks(self) -> List[Task]:
        """Generate tasks with varying parameters."""
        tasks = []
        
        for i in range(self.n_tasks):
            params = {}
            difficulty = i / (self.n_tasks - 1) if self.n_tasks > 1 else 0.5
            
            # Track if we're varying speed range
            speed_range_low = None
            speed_range_high = None
            
            for param in self.vary_params:
                if param in self.param_ranges:
                    low, high = self.param_ranges[param]
                    # Use difficulty to interpolate parameter values
                    value = low + difficulty * (high - low)
                    
                    # Add some randomness
                    noise_scale = 0.1 * (high - low)
                    value += self.rng.normal(0, noise_scale)
                    value = np.clip(value, low, high)
                    
                    # Convert to int if needed
                    if param in ["lanes_count", "spots_available", "duration"]:
                        value = int(value)
                    
                    # Handle speed range parameters
                    if param == "reward_speed_range_low":
                        speed_range_low = value
                    elif param == "reward_speed_range_high":
                        speed_range_high = value
                    else:
                        params[param] = value
            
            # If we have both speed range values, create the list
            if speed_range_low is not None and speed_range_high is not None:
                # Ensure low < high
                if speed_range_low >= speed_range_high:
                    speed_range_low, speed_range_high = speed_range_high - 5, speed_range_high
                params["reward_speed_range"] = [speed_range_low, speed_range_high]
            
            task = Task(
                task_id=i,
                params=params,
                description=f"{self.env_type}_task_{i}",
                difficulty=difficulty
            )
            tasks.append(task)
        
        return tasks
    
    def sample_task(self) -> Task:
        """Sample a random task."""
        idx = self.rng.integers(0, len(self.tasks))
        return self.tasks[idx]

    def sample_tasks(self, n: int) -> List[Task]:
        """Sample n tasks without replacement."""
        if n > len(self.tasks):
            raise ValueError(f"Cannot sample {n} tasks from {len(self.tasks)} tasks")
        indices = self.rng.choice(len(self.tasks), size=n, replace=False)
        return [self.tasks[i] for i in indices]


# =================================================================
#  CausalTaskDistribution — latent-confounder SCM for causal experiments
# =================================================================
#
#  Structural Causal Model (SCM)
#  ─────────────────────────────
#  This distribution introduces a *latent confounder* that is the
#  common cause of several observable task parameters.  The causal
#  graph is:
#
#      weather (latent, U ∈ [0, 1])
#        │
#        ├──→ speed_limit          (lower when weather is bad)
#        ├──→ driver_aggressiveness (higher when weather is bad)
#        └──→ vehicles_density     (lower when weather is bad)
#
#  "weather" represents road conditions on a continuous scale:
#      0.0 = clear / dry   →  high speed limit, calm drivers, dense traffic
#      0.5 = rain           →  moderate limits, moderate aggression
#      1.0 = fog / ice      →  low speed limit, aggressive (panicky) drivers,
#                               sparse traffic
#
#  The confounder is stored in task.params with an underscore prefix
#  ("_confounder_weather") so that MetaHighwayEnv._configure_env_for_task()
#  does NOT inject it into the environment config.  It is exposed only
#  through info["task"]["confounder"] for post-hoc causal analysis.
#
#  Why this matters for C2A
#  ~~~~~~~~~~~~~~~~~~~~~~~~
#  Without observing the confounder, an agent that naively correlates
#  speed_limit with reward may confuse the *direct* effect of speed_limit
#  with the *confounded* effect mediated by weather → aggressiveness.
#  C2A's causal inference module should be able to discover this
#  structure and adjust its context representation accordingly.
# =================================================================


class CausalTaskDistribution(TaskDistribution):
    """Task distribution with a latent confounder generating correlated task params.

    The latent variable ``weather`` ∈ [0, 1] acts as a common cause of
    ``speed_limit``, ``driver_aggressiveness``, and ``vehicles_density``.
    The agent never observes ``weather`` directly; it must be inferred
    from the observable dynamics (or accessed via ``info`` for analysis).

    Causal graph::

        weather (latent)
          ├──→ speed_limit           = f₁(weather) + ε₁
          ├──→ driver_aggressiveness  = f₂(weather) + ε₂
          └──→ vehicles_density       = f₃(weather) + ε₃

    Structural equations (default, configurable via ``causal_coefficients``):

        speed_limit           = 30 - 15 * weather + ε₁     ∈ [15, 30]
        driver_aggressiveness = 0.1 + 0.8 * weather + ε₂   ∈ [0, 1]
        vehicles_density      = 2.0 - 1.5 * weather + ε₃   ∈ [0.5, 2.0]

    Parameters
    ----------
    n_tasks : int
        Number of tasks to generate.
    noise_scale : float
        Standard deviation of additive Gaussian noise on each structural
        equation (default 0.05).
    confounder_distribution : str
        How to sample the confounder: ``"uniform"`` (default) or
        ``"discrete"`` (samples from {0.0, 0.5, 1.0} = clear/rain/fog).
    causal_coefficients : dict, optional
        Override the linear SCM coefficients.  Keys are
        ``"speed_limit"``, ``"driver_aggressiveness"``,
        ``"vehicles_density"``, each mapping to ``(intercept, slope)``.
    extra_params : dict, optional
        Additional fixed parameters to include in every task
        (e.g. ``{"lanes_count": 4, "collision_reward": -1.0}``).
    seed : int, optional
        Random seed for reproducibility.
    """

    # Default structural equations: (intercept, slope, clip_low, clip_high)
    _DEFAULT_SCM: Dict[str, Tuple[float, float, float, float]] = {
        #                    intercept  slope  clip_lo  clip_hi
        "speed_limit":           (30.0, -15.0,  15.0,   30.0),
        "driver_aggressiveness": ( 0.1,   0.8,   0.0,    1.0),
        "vehicles_density":      ( 2.0,  -1.5,   0.5,    2.0),
    }

    def __init__(
        self,
        n_tasks: int,
        noise_scale: float = 0.05,
        confounder_distribution: str = "uniform",
        causal_coefficients: Optional[Dict[str, Tuple[float, float]]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.noise_scale = noise_scale
        self.confounder_distribution = confounder_distribution
        self.extra_params = extra_params or {}

        # Build SCM table, allowing partial overrides
        self.scm: Dict[str, Tuple[float, float, float, float]] = dict(
            self._DEFAULT_SCM
        )
        if causal_coefficients is not None:
            for var, (intercept, slope) in causal_coefficients.items():
                if var in self.scm:
                    _, _, lo, hi = self.scm[var]
                    self.scm[var] = (intercept, slope, lo, hi)

        super().__init__(n_tasks, seed)

    # ---- confounder sampling -------------------------------------------

    def _sample_confounder(self) -> float:
        """Sample the latent confounder ``weather``."""
        if self.confounder_distribution == "discrete":
            return float(self.rng.choice([0.0, 0.5, 1.0]))
        else:  # "uniform"
            return float(self.rng.uniform(0.0, 1.0))  # default_rng.uniform is identical

    # ---- structural equations ------------------------------------------

    def _structural_equation(
        self, weather: float, variable: str
    ) -> float:
        """Evaluate one structural equation: X_i = f_i(weather) + ε_i."""
        intercept, slope, lo, hi = self.scm[variable]
        noise = self.rng.normal(0.0, self.noise_scale * (hi - lo))  # default_rng.normal is identical
        value = intercept + slope * weather + noise
        return float(np.clip(value, lo, hi))

    # ---- task generation -----------------------------------------------

    def _generate_tasks(self) -> List[Task]:
        """Generate tasks from the latent-confounder SCM."""
        tasks = []
        for i in range(self.n_tasks):
            weather = self._sample_confounder()

            params: Dict[str, Any] = {}

            # Structural equations: weather → observable params
            for variable in self.scm:
                params[variable] = self._structural_equation(weather, variable)

            # Store confounder with underscore prefix (hidden from env config)
            params["_confounder_weather"] = weather

            # Merge any fixed extra parameters
            params.update(self.extra_params)

            # Difficulty = weather (bad weather ≈ harder)
            task = Task(
                task_id=i,
                params=params,
                description=(
                    f"causal_task_{i}_weather={weather:.2f}"
                ),
                difficulty=weather,
            )
            tasks.append(task)
        return tasks

    def sample_task(self) -> Task:
        """Sample a random task from the pre-generated set."""
        idx = self.rng.integers(0, len(self.tasks))
        return self.tasks[idx]

    def sample_fresh_task(self, task_id: Optional[int] = None) -> Task:
        """Sample a *new* task on the fly (not from the fixed set).

        Useful for evaluation where you want an unseen confounder value.
        """
        weather = self._sample_confounder()
        params: Dict[str, Any] = {}
        for variable in self.scm:
            params[variable] = self._structural_equation(weather, variable)
        params["_confounder_weather"] = weather
        params.update(self.extra_params)

        tid = task_id if task_id is not None else -1
        return Task(
            task_id=tid,
            params=params,
            description=f"causal_fresh_weather={weather:.2f}",
            difficulty=weather,
        )

    def get_causal_graph_description(self) -> str:
        """Return a human-readable description of the SCM."""
        lines = ["Causal Graph (SCM):", "  weather (latent) ∈ [0, 1]"]
        for var, (intercept, slope, lo, hi) in self.scm.items():
            sign = "+" if slope >= 0 else ""
            lines.append(
                f"    └──→ {var} = {intercept}{sign}{slope}*weather + ε  "
                f"∈ [{lo}, {hi}]"
            )
        return "\n".join(lines)

    @classmethod
    def from_scm_graph(
        cls,
        scm_graph,
        n_tasks: int,
        extra_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> "CausalTaskDistribution":
        """Create a CausalTaskDistribution from an :class:`SCMGraph`.

        This factory bridges the declarative :class:`~highway_env.meta_rl.scm.SCMGraph`
        engine with the task-distribution system.  Each task is generated by
        calling ``scm_graph.sample()``, with latent nodes stored as
        underscore-prefixed hidden params and observed nodes as regular
        task params.

        Args:
            scm_graph:    An :class:`~highway_env.meta_rl.scm.SCMGraph` instance.
            n_tasks:      Number of tasks to generate.
            extra_params: Additional fixed params for every task.
            seed:         Random seed (overrides scm_graph's rng).

        Returns:
            A ``CausalTaskDistribution`` whose tasks were sampled from
            the graph.

        Example::

            from highway_env.meta_rl.scm import SCMGraph

            scm = SCMGraph.default_highway_scm_with_iv(seed=42)
            dist = CausalTaskDistribution.from_scm_graph(scm, n_tasks=50)
        """
        # We bypass __init__ and manually generate tasks from the graph
        obj = cls.__new__(cls)

        # Minimal init to satisfy TaskDistribution contract
        obj.n_tasks = n_tasks
        obj.rng = np.random.default_rng(seed)
        obj.noise_scale = scm_graph.noise_scale
        obj.confounder_distribution = "graph"
        obj.extra_params = extra_params or {}
        obj.scm = {}  # not used when sourced from graph
        obj._scm_graph = scm_graph

        if seed is not None:
            scm_graph.rng = np.random.default_rng(seed)

        tasks = []
        latent_names = scm_graph.latent_nodes()
        for i in range(n_tasks):
            sample = scm_graph.sample()

            params: Dict[str, Any] = {}
            difficulty_vals = []
            for name, value in sample.items():
                node = scm_graph._nodes[name]
                if node.node_type == "latent":
                    params[f"_{name}"] = value
                    difficulty_vals.append(value)
                else:
                    params[name] = value

            params.update(obj.extra_params)

            difficulty = float(np.mean(difficulty_vals)) if difficulty_vals else 0.5
            confounder_desc = ", ".join(
                f"{n}={sample[n]:.2f}" for n in latent_names
            )
            tasks.append(Task(
                task_id=i,
                params=params,
                description=f"scm_task_{i}_{confounder_desc}",
                difficulty=difficulty,
            ))

        obj.tasks = tasks
        return obj


class MultiEnvTaskDistribution(TaskDistribution):
    """Task distribution spanning multiple Highway environment types."""
    
    def __init__(self,
                 n_tasks: int,
                 env_types: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        
        self.env_types = env_types or ["highway", "intersection", "merge", "roundabout"]
        super().__init__(n_tasks, seed)
    
    def _generate_tasks(self) -> List[Task]:
        """Generate tasks across different environment types."""
        tasks = []
        tasks_per_env = self.n_tasks // len(self.env_types)
        remaining = self.n_tasks % len(self.env_types)
        
        task_id = 0
        for i, env_type in enumerate(self.env_types):
            n_env_tasks = tasks_per_env + (1 if i < remaining else 0)
            
            # Create sub-distribution for this env type
            sub_dist = HighwayTaskDistribution(
                n_tasks=n_env_tasks,
                env_type=env_type,
                seed=int(self.rng.integers(0, 10000))
            )
            
            for task in sub_dist.get_all_tasks():
                # Update task ID and description
                task.task_id = task_id
                task.description = f"multi_{env_type}_task_{task_id}"
                task.params["env_type"] = env_type
                tasks.append(task)
                task_id += 1
        
        return tasks
    
    def sample_task(self) -> Task:
        """Sample a random task."""
        idx = self.rng.integers(0, len(self.tasks))
        return self.tasks[idx]

    def sample_env_type_task(self, env_type: str) -> Task:
        """Sample a task from a specific environment type."""
        env_tasks = [t for t in self.tasks if t.params.get("env_type") == env_type]
        if not env_tasks:
            raise ValueError(f"No tasks found for environment type: {env_type}")
        idx = self.rng.integers(0, len(env_tasks))
        return env_tasks[idx]