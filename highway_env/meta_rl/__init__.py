"""Meta-RL components for Highway environments."""

from .task_distribution import (
    TaskDistribution,
    HighwayTaskDistribution,
    CausalTaskDistribution,
    MultiEnvTaskDistribution,
    Task,
)

TaskSpec = Task
from .meta_env import MetaHighwayEnv, get_normalized_score
from .task_sampler import TaskSampler, UniformTaskSampler, CurriculumTaskSampler, AdaptiveTaskSampler
from .counterfactual import CounterfactualEngine, SimulatorState
from .scm import SCMGraph, SCMNode, SCMEdge
from .causal_graph import (
    WITHIN_EPISODE_DAG,
    TASK_LEVEL_DAG,
    dag_to_dot,
    get_full_two_level_dag,
)

__all__ = [
    # Task distributions
    "TaskDistribution",
    "HighwayTaskDistribution",
    "CausalTaskDistribution",
    "MultiEnvTaskDistribution",
    "Task",
    "TaskSpec",
    # Meta-RL env
    "MetaHighwayEnv",
    "get_normalized_score",
    # Task sampling
    "TaskSampler",
    "UniformTaskSampler",
    "CurriculumTaskSampler",
    "AdaptiveTaskSampler",
    # Counterfactual engine (Gap 1)
    "CounterfactualEngine",
    "SimulatorState",
    # SCM graph engine (Gap 2)
    "SCMGraph",
    "SCMNode",
    "SCMEdge",
    # Causal DAG documentation (Gap 4)
    "WITHIN_EPISODE_DAG",
    "TASK_LEVEL_DAG",
    "dag_to_dot",
    "get_full_two_level_dag",
]
