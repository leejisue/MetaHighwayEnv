"""Meta-RL wrapper for Highway environments."""

import warnings

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from gymnasium import spaces

from ..envs import (
    ExitEnv,
    HighwayEnv,
    HighwayEnvFast,
    IntersectionEnv,
    ContinuousIntersectionEnv,
    MultiAgentIntersectionEnv,
    LaneKeepingEnv,
    MergeEnv,
    RacetrackEnv,
    RacetrackEnvLarge,
    RacetrackEnvOval,
    RoundaboutEnv,
    TwoWayEnv,
    UTurnEnv,
)
from .task_distribution import Task, TaskDistribution, HighwayTaskDistribution
from .task_sampler import TaskSampler, UniformTaskSampler


class MetaHighwayEnv(gym.Env):
    """Meta-RL environment wrapper for Highway environments.

    This wrapper manages task sampling, context collection, and
    environment configuration for meta-learning algorithms.

    All registered gymnasium environment types are supported as
    ``env_type`` values.  Each one includes the full Causal
    Intervention API via :class:`CausalInterventionMixin`.
    """

    # D4RL-style reference returns: {env_type: {task_name: (random, expert)}}
    REF_RETURNS: Dict[str, Dict[str, Tuple[float, float]]] = {
        "highway": {
            "easy": (14.62, 27.34), "dense": (9.35, 25.87),
            "fast": (12.62, 27.39), "dangerous": (12.38, 32.02),
            "slow_dense": (7.79, 19.43), "mixed": (13.27, 30.48),
        },
        "intersection": {
            "default": (0.05, 3.94), "dense": (-2.26, 2.19),
            "sparse": (5.56, 7.05), "fast": (6.09, 6.35),
            "dangerous": (-5.39, 0.26), "slow": (0.11, 4.96),
        },
        "merge": {
            "default": (14.25, 24.93), "dense": (11.70, 19.69),
            "sparse": (15.69, 24.38), "fast": (11.09, 24.33),
            "dangerous": (16.03, 29.70), "calm": (11.19, 18.87),
        },
        "roundabout": {
            "default": (0.87, 1.43), "dense": (0.52, 0.71),
            "sparse": (2.14, 9.69), "fast": (1.02, 1.93),
            "dangerous": (0.52, 0.65), "slow": (0.57, 1.16),
        },
        "racetrack": {
            "default": (4.12, 112.54), "wide": (6.10, 120.22),
            "narrow": (0.43, 0.54), "fast": (2.27, 108.84),
            "slow": (7.96, 127.06), "crowded": (4.12, 48.53),
        },
    }

    ENV_CLASSES = {
        # ---- Core environment types ----
        "highway": HighwayEnv,
        "intersection": IntersectionEnv,
        "merge": MergeEnv,
        "roundabout": RoundaboutEnv,
        "racetrack": RacetrackEnv,
        # ---- Variant / extended environment types ----
        "exit": ExitEnv,
        "highway-fast": HighwayEnvFast,
        "intersection-continuous": ContinuousIntersectionEnv,
        "intersection-multi-agent": MultiAgentIntersectionEnv,
        "lane-keeping": LaneKeepingEnv,
        "racetrack-large": RacetrackEnvLarge,
        "racetrack-oval": RacetrackEnvOval,
        "two-way": TwoWayEnv,
        "u-turn": UTurnEnv,
    }
    
    def __init__(self,
                 env_type: str = "highway",
                 task_distribution: Optional[TaskDistribution] = None,
                 task_sampler: Optional[TaskSampler] = None,
                 n_tasks: int = 10,
                 context_length: int = 100,
                 include_task_id: bool = False,
                 enable_causal_logging: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            env_type: Type of highway environment
            task_distribution: Task distribution to use (creates default if None)
            task_sampler: Task sampling strategy (creates uniform if None)
            n_tasks: Number of tasks in distribution (if creating default)
            context_length: Length of context episodes for meta-learning
            include_task_id: Whether to include task ID in observation
            enable_causal_logging: Whether to expose task/confounder metadata
                in ``info["task"]`` on every step and record intervention
                history.  Disable (default) for fast headless training;
                enable for causal analysis and debugging.
            seed: Random seed
        """
        super().__init__()

        self.env_type = env_type
        self.context_length = context_length
        self.include_task_id = include_task_id
        self.enable_causal_logging = enable_causal_logging
        self._init_seed = seed

        # Initialize task distribution
        if task_distribution is None:
            self.task_distribution = HighwayTaskDistribution(
                n_tasks=n_tasks,
                env_type=env_type,
                seed=seed
            )
        else:
            self.task_distribution = task_distribution
        
        # Initialize task sampler
        if task_sampler is None:
            self.task_sampler = UniformTaskSampler(
                self.task_distribution,
                seed=seed
            )
        else:
            self.task_sampler = task_sampler
        
        # Create base environment for space definitions
        self._create_base_env()

        # All environment types now support the causal intervention API
        # via CausalInterventionMixin. Verify at runtime as a safety check.
        if self.enable_causal_logging and not hasattr(self.base_env, "intervene"):
            warnings.warn(
                f"Base environment {type(self.base_env).__name__} does not "
                f"have an intervene() method.  This is unexpected — all "
                f"bundled environment types should include the "
                f"CausalInterventionMixin.  Causal intervention calls "
                f"will raise AttributeError.",
                UserWarning,
                stacklevel=2,
            )
        
        # Current task and episode tracking
        self.current_task: Optional[Task] = None
        self.episode_count = 0
        self.context_episodes = []
        self.is_meta_testing = False
        
        # Performance tracking
        self.task_performances = {}

        # Persistent intervention history (survives reset; see Gap 5)
        self._full_intervention_history: List[Dict[str, Any]] = []
    
    def _create_base_env(self) -> None:
        """Create base environment instance for getting spaces."""
        env_class = self.ENV_CLASSES.get(self.env_type)
        if env_class is None:
            raise ValueError(f"Unknown environment type: {self.env_type}")
        
        self.base_env = env_class()
        
        # Define observation space (potentially with task ID)
        base_obs_space = self.base_env.observation_space
        if self.include_task_id:
            # Add task ID to observation
            if isinstance(base_obs_space, spaces.Box):
                # Flatten if needed
                if len(base_obs_space.shape) > 1:
                    flat_dim = np.prod(base_obs_space.shape)
                    low = np.concatenate([base_obs_space.low.flatten(), [0]])
                    high = np.concatenate([base_obs_space.high.flatten(), [self.task_distribution.n_tasks]])
                else:
                    low = np.concatenate([base_obs_space.low, [0]])
                    high = np.concatenate([base_obs_space.high, [self.task_distribution.n_tasks]])
                self.observation_space = spaces.Box(low=low, high=high, dtype=base_obs_space.dtype)
            else:
                # For other space types, create a dict space
                self.observation_space = spaces.Dict({
                    "obs": base_obs_space,
                    "task_id": spaces.Discrete(self.task_distribution.n_tasks)
                })
        else:
            self.observation_space = base_obs_space
        
        self.action_space = self.base_env.action_space
    
    def sample_task(self) -> Task:
        """Sample a new task."""
        tasks = self.task_sampler.sample(n_tasks=1)
        return tasks[0]
    
    def set_task(self, task: Union[Task, int]) -> None:
        """Set the current task.
        
        Args:
            task: Task object or task ID
        """
        if isinstance(task, int):
            task = self.task_distribution.get_task(task)
        
        self.current_task = task
        self._configure_env_for_task()
        self.episode_count = 0
        self.context_episodes = []
    
    def _configure_env_for_task(self) -> None:
        """Configure environment parameters for current task.

        Parameters whose name starts with ``_`` (e.g. ``_confounder_weather``)
        are **hidden** — they are stored in the Task but never injected into
        the base environment config.  This is how latent confounders stay
        invisible to the agent's observation while remaining available for
        post-hoc causal analysis via ``info["task"]``.
        """
        if self.current_task is None:
            return

        # Update environment config
        config = self.base_env.config.copy()

        # Backward-compat: rename legacy arrival_probability → spawn_probability
        if "arrival_probability" in config:
            warnings.warn(
                "arrival_probability is deprecated, use spawn_probability",
                DeprecationWarning,
                stacklevel=2,
            )
            config["spawn_probability"] = config.pop("arrival_probability")

        # Backward-compat: rename legacy arrival_probability in task params
        if "arrival_probability" in self.current_task.params:
            warnings.warn(
                "Task param arrival_probability is deprecated, use spawn_probability",
                DeprecationWarning,
                stacklevel=2,
            )
            self.current_task.params["spawn_probability"] = self.current_task.params.pop("arrival_probability")

        # Apply task parameters (skip underscore-prefixed hidden params)
        for param, value in self.current_task.params.items():
            if param.startswith("_"):
                # Hidden / latent variable — do NOT inject into env config
                continue
            if param in config:
                config[param] = value
            elif "_" in param:
                # Handle nested parameters (e.g., "reward_speed_range")
                parts = param.split("_", 1)
                if parts[0] in config and isinstance(config[parts[0]], dict):
                    config[parts[0]][parts[1]] = value

        self.base_env.configure(config)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment, potentially sampling a new task."""
        # Seed numpy rng on first reset or when explicitly provided
        _seed = seed if seed is not None else self._init_seed
        if _seed is not None:
            self._np_rng = np.random.default_rng(_seed)
            self._init_seed = None  # consume once

        # Sample new task if needed
        if self.current_task is None or (
            not self.is_meta_testing and
            self.episode_count >= self.context_length
        ):
            self.set_task(self.sample_task())

        # Clear per-episode intervention log so it doesn't grow unbounded
        self._intervention_log = []

        # Reset base environment using gymnasium API (seed forwarded)
        reset_kwargs: Dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options
        obs, info = self.base_env.reset(**reset_kwargs)

        # Add task ID if needed
        if self.include_task_id:
            if isinstance(self.observation_space, spaces.Box):
                if len(obs.shape) > 1:
                    obs = obs.flatten()
                obs = np.concatenate([obs, [self.current_task.task_id]])
            else:
                obs = {"obs": obs, "task_id": self.current_task.task_id}

        self.episode_count += 1
        self.current_episode_data = {
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "dones": [],
            "task_id": self.current_task.task_id
        }

        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action in environment."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        done = terminated or truncated

        # Add task ID if needed
        if self.include_task_id:
            if isinstance(self.observation_space, spaces.Box):
                if len(obs.shape) > 1:
                    obs = obs.flatten()
                obs = np.concatenate([obs, [self.current_task.task_id]])
            else:
                obs = {"obs": obs, "task_id": self.current_task.task_id}

        # Expose task metadata in info (gated for performance)
        if self.enable_causal_logging and self.current_task is not None:
            task_info: Dict[str, Any] = {
                "task_id": self.current_task.task_id,
                "difficulty": self.current_task.difficulty,
            }
            confounders = {
                k.lstrip("_"): v
                for k, v in self.current_task.params.items()
                if k.startswith("_")
            }
            if confounders:
                task_info["confounder"] = confounders
            info["task"] = task_info

        # Store transition data
        self.current_episode_data["observations"].append(obs)
        self.current_episode_data["actions"].append(action)
        self.current_episode_data["rewards"].append(reward)
        self.current_episode_data["dones"].append(done)

        if done:
            self.context_episodes.append(self.current_episode_data)
            episode_return = sum(self.current_episode_data["rewards"])
            task_id = self.current_task.task_id
            if task_id not in self.task_performances:
                self.task_performances[task_id] = []
            self.task_performances[task_id].append(episode_return)
            avg_performance = np.mean(self.task_performances[task_id][-10:])
            self.task_sampler.update({task_id: avg_performance})

        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.base_env.render()

    def close(self) -> None:
        """Close the environment."""
        self.base_env.close()
    
    def get_context(self, n_episodes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get context episodes for current task.
        
        Args:
            n_episodes: Number of recent episodes to return (None for all)
            
        Returns:
            List of episode data dictionaries
        """
        if n_episodes is None:
            return self.context_episodes.copy()
        else:
            return self.context_episodes[-n_episodes:].copy()
    
    def enable_meta_testing(self) -> None:
        """Enable meta-testing mode (no task switching)."""
        self.is_meta_testing = True
    
    def disable_meta_testing(self) -> None:
        """Disable meta-testing mode."""
        self.is_meta_testing = False
    
    # =================================================================
    #  Causal Intervention API — delegates to base_env.intervene()
    # =================================================================

    def intervene(
        self,
        variable: str,
        value,
    ) -> Dict[str, Any]:
        """Perform do(variable := value) on the current task's environment.

        Delegates to ``base_env.intervene()`` (see
        :class:`~highway_env.envs.highway_env.HighwayEnv` for the full
        variable list and granularity semantics).

        The intervention metadata is also recorded in the current task's
        context so that causal analysis can track what was intervened on.

        Args:
            variable: Causal variable name (e.g. ``"driver_aggressiveness"``).
            value:    Interventional value.

        Returns:
            Metadata dict from :meth:`HighwayEnv.intervene`.
        """
        meta = self.base_env.intervene(variable, value)

        # Record intervention in logs only when causal logging is enabled.
        if self.enable_causal_logging:
            if not hasattr(self, "_intervention_log"):
                self._intervention_log = []
            entry = {
                "episode": getattr(self, "episode_count", 0),
                "task_id": self.current_task.task_id if self.current_task else None,
                **meta,
            }
            self._intervention_log.append(entry)

            # Also record in persistent history (never auto-cleared)
            if not hasattr(self, "_full_intervention_history"):
                self._full_intervention_history = []
            self._full_intervention_history.append(entry)

        return meta

    def intervene_batch(
        self,
        interventions: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Apply multiple interventions at once.

        See :meth:`HighwayEnv.intervene_batch`.
        """
        return {var: self.intervene(var, val) for var, val in interventions.items()}

    def interventionable_variables(self) -> Dict[str, str]:
        """Return variable name → granularity mapping for the current env.

        Delegates to the base environment's
        :meth:`CausalInterventionMixin.interventionable_variables`.
        """
        if hasattr(self.base_env, "interventionable_variables"):
            return self.base_env.interventionable_variables()
        # Fallback for any non-mixin env (should not happen with bundled envs)
        return {}

    def get_causal_state(self) -> Dict[str, Any]:
        """Return the structured causal state of the base environment.

        Delegates to :meth:`CausalInterventionMixin.get_causal_state`.
        """
        if hasattr(self.base_env, "get_causal_state"):
            state = self.base_env.get_causal_state()
        else:
            state = {}
        # Enrich with meta-RL context
        if self.current_task is not None:
            state["task_id"] = self.current_task.task_id
            state["task_difficulty"] = self.current_task.difficulty
        state["env_type"] = self.env_type
        return state

    def decompose_reward(self, info: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
        """Return decomposed reward components from the last step.

        Delegates to :meth:`CausalInterventionMixin.decompose_reward`.
        """
        if hasattr(self.base_env, "decompose_reward"):
            return self.base_env.decompose_reward(info)
        return {}

    def get_intervention_log(self) -> List[Dict[str, Any]]:
        """Return the list of all interventions applied in this session."""
        return getattr(self, "_intervention_log", []).copy()

    def clear_intervention_log(self) -> None:
        """Clear the per-session intervention log (does NOT clear history)."""
        self._intervention_log = []

    def get_full_intervention_history(self) -> List[Dict[str, Any]]:
        """Return the persistent intervention history across all episodes.

        Unlike :meth:`get_intervention_log`, this history is never
        automatically cleared.  Each entry is tagged with ``episode``
        and ``task_id`` for cross-episode causal analysis.
        """
        return getattr(self, "_full_intervention_history", []).copy()

    def clear_full_intervention_history(self) -> None:
        """Explicitly clear the persistent intervention history."""
        self._full_intervention_history = []

    def get_task_performance(self, task_id: Optional[int] = None) -> Dict[int, float]:
        """Get average performance for tasks.
        
        Args:
            task_id: Specific task ID (None for all tasks)
            
        Returns:
            Dictionary mapping task IDs to average returns
        """
        if task_id is not None:
            if task_id in self.task_performances:
                return {task_id: np.mean(self.task_performances[task_id])}
            else:
                return {task_id: 0.0}
        else:
            return {
                tid: np.mean(perfs)
                for tid, perfs in self.task_performances.items()
            }

    # =================================================================
    #  D4RL-style Normalized Score
    # =================================================================

    def get_normalized_score(self, raw_return: float,
                             task_name: Optional[str] = None) -> float:
        """Compute D4RL-style normalized score for a raw episodic return.

        .. math::

            \\text{score} = \\frac{\\text{return} - \\text{return}_{\\text{random}}}
                                  {\\text{return}_{\\text{expert}} - \\text{return}_{\\text{random}}}
                           \\times 100

        A score of 0 corresponds to random-policy performance and 100 to
        expert-policy performance.  Scores above 100 indicate super-expert
        behaviour.

        If *task_name* is ``None``, the environment-level average of random
        and expert returns is used (suitable when the task variant is
        unknown or when reporting aggregate metrics).

        Args:
            raw_return: Raw episodic return to normalize.
            task_name:  Task variant name (e.g. ``"dense"``).  If ``None``,
                        uses the env-level mean reference returns.

        Returns:
            Normalized score (float).  Returns ``float('nan')`` when
            reference data is unavailable.
        """
        env_refs = self.REF_RETURNS.get(self.env_type)
        if env_refs is None:
            return float("nan")

        if task_name is not None and task_name in env_refs:
            rand_ret, expert_ret = env_refs[task_name]
        else:
            # Average across all tasks in this env
            rands = [v[0] for v in env_refs.values()]
            experts = [v[1] for v in env_refs.values()]
            rand_ret = float(np.mean(rands))
            expert_ret = float(np.mean(experts))

        denom = expert_ret - rand_ret
        if abs(denom) < 1e-8:
            return float("nan")

        return (raw_return - rand_ret) / denom * 100.0

    @classmethod
    def get_ref_returns(cls, env_type: str) -> Dict[str, Tuple[float, float]]:
        """Return the reference (random, expert) returns for an env type.

        Args:
            env_type: Environment type name (e.g. ``"highway"``).

        Returns:
            Dict mapping task names to ``(random_return, expert_return)``
            tuples.  Empty dict if *env_type* is unknown.
        """
        return cls.REF_RETURNS.get(env_type, {})


def get_normalized_score(raw_score: float, env_name: str) -> float:
    """Normalize raw return to [0, 1] using env-specific reference range.

    Convenience wrapper around :meth:`MetaHighwayEnv.get_normalized_score`
    for use without an env instance.

    Args:
        raw_score: Raw episodic return to normalize.
        env_name:  Environment name (e.g. ``"highway"``).

    Returns:
        Normalized score in approximately [0, 1] (100 = expert level).
        Returns ``float('nan')`` when reference data is unavailable.

    Example::

        from highway_env.meta_rl import get_normalized_score
        score = get_normalized_score(10.0, env_name="highway")
    """
    env_refs = MetaHighwayEnv.REF_RETURNS.get(env_name)
    if env_refs is None:
        return float("nan")
    rands = [v[0] for v in env_refs.values()]
    experts = [v[1] for v in env_refs.values()]
    rand_ret = float(np.mean(rands))
    expert_ret = float(np.mean(experts))
    denom = expert_ret - rand_ret
    if abs(denom) < 1e-8:
        return float("nan")
    return (raw_score - rand_ret) / denom * 100.0