"""Within-episode causal DAG specification for the Highway environment.

This module formally documents the causal structure that operates
*within* a single episode — i.e. the per-timestep data-generating
process (DGP) of the highway simulator.

Two DAGs are provided:

1. :data:`WITHIN_EPISODE_DAG` — full per-step causal graph including
   dynamics, reward, and observation generation.
2. :data:`TASK_LEVEL_DAG` — the between-episode SCM describing how
   latent confounders generate task parameters.

Together they define the *two-level* causal structure that C2A's
context encoder must disentangle:

    Task-level:   weather → {speed_limit, aggressiveness, density}
    Step-level:   action → ego_state → traffic_response → reward → obs

These DAGs are represented as plain dicts so they can be:
- Programmatically inspected by experiment scripts
- Exported to DOT/Graphviz for paper figures
- Compared against learned causal graphs for evaluation
"""

from __future__ import annotations
from typing import Dict, List, Any


# ======================================================================
#  Within-episode (per-timestep) causal DAG
# ======================================================================

WITHIN_EPISODE_DAG: Dict[str, Any] = {
    "description": (
        "Per-timestep causal graph of the highway driving simulator. "
        "Subscript _t denotes the current timestep."
    ),

    "nodes": [
        # ── Agent decision ──
        {"name": "action_t",            "type": "decision",
         "description": "Discrete action chosen by agent (LANE_LEFT/IDLE/LANE_RIGHT/FASTER/SLOWER)"},

        # ── Ego vehicle state ──
        {"name": "ego_speed_t",         "type": "state",
         "description": "Ego vehicle forward speed after action execution"},
        {"name": "ego_position_t",      "type": "state",
         "description": "Ego vehicle (x, y) position on road"},
        {"name": "ego_lane_t",          "type": "state",
         "description": "Ego vehicle current lane index"},

        # ── Traffic (other vehicles) ──
        {"name": "traffic_state_t",     "type": "state",
         "description": "Positions, speeds, lanes of all non-ego IDMVehicles"},

        # ── Task-level params (constant within episode) ──
        {"name": "driver_aggressiveness", "type": "task_param",
         "description": "IDM/MOBIL aggressiveness of traffic vehicles (set at task creation)"},
        {"name": "speed_limit",          "type": "task_param",
         "description": "Lane speed limit (affects IDM desired speed)"},
        {"name": "vehicles_density",     "type": "task_param",
         "description": "Traffic density (set at episode start via reset)"},

        # ── Reward components ──
        {"name": "collision_t",         "type": "event",
         "description": "Binary: ego collided with another vehicle this step"},
        {"name": "speed_reward_t",      "type": "reward_component",
         "description": "Scaled speed reward ∈ [0, 1], linearly mapped from reward_speed_range"},
        {"name": "right_lane_reward_t", "type": "reward_component",
         "description": "Normalised lane index (rightmost = 1)"},
        {"name": "lane_change_reward_t", "type": "reward_component",
         "description": "Binary indicator: did agent change lanes this step"},

        # ── Aggregate ──
        {"name": "reward_t",           "type": "reward",
         "description": "Weighted sum of reward components"},
        {"name": "observation_t1",     "type": "observation",
         "description": "Kinematics matrix (5×5) presented to agent at t+1"},
    ],

    "edges": [
        # Action → ego state
        ("action_t",              "ego_speed_t"),
        ("action_t",              "ego_lane_t"),

        # Ego state dynamics
        ("ego_speed_t",           "ego_position_t"),

        # Ego affects traffic (IDM following distance reaction)
        ("ego_position_t",        "traffic_state_t"),
        ("ego_speed_t",           "traffic_state_t"),

        # Task params affect traffic behaviour
        ("driver_aggressiveness", "traffic_state_t"),
        ("speed_limit",           "traffic_state_t"),
        ("vehicles_density",      "traffic_state_t"),

        # Speed limit also constrains ego's effective speed reward
        ("speed_limit",           "speed_reward_t"),

        # Traffic state → collision
        ("traffic_state_t",       "collision_t"),
        ("ego_position_t",        "collision_t"),
        ("ego_speed_t",           "collision_t"),

        # State → reward components
        ("ego_speed_t",           "speed_reward_t"),
        ("ego_lane_t",            "right_lane_reward_t"),
        ("action_t",              "lane_change_reward_t"),
        ("collision_t",           "reward_t"),

        # Reward components → total reward
        ("speed_reward_t",        "reward_t"),
        ("right_lane_reward_t",   "reward_t"),
        ("lane_change_reward_t",  "reward_t"),

        # State → next observation
        ("traffic_state_t",       "observation_t1"),
        ("ego_position_t",        "observation_t1"),
        ("ego_speed_t",           "observation_t1"),
        ("ego_lane_t",            "observation_t1"),
    ],
}


# ======================================================================
#  Task-level (between-episode) causal DAG
# ======================================================================

TASK_LEVEL_DAG: Dict[str, Any] = {
    "description": (
        "Between-episode causal graph: how latent confounders "
        "generate task parameters that remain constant within an episode."
    ),

    "nodes": [
        {"name": "weather",              "type": "latent_confounder",
         "description": "Latent road condition ∈ [0,1] (0=clear, 1=fog/ice)"},
        {"name": "road_construction",    "type": "instrumental_variable",
         "description": "IV ∈ [0,1], independent of weather, affects density only"},
        {"name": "speed_limit",          "type": "observed_task_param"},
        {"name": "driver_aggressiveness", "type": "observed_task_param"},
        {"name": "vehicles_density",     "type": "observed_task_param"},
    ],

    "edges": [
        ("weather", "speed_limit"),
        ("weather", "driver_aggressiveness"),
        ("weather", "vehicles_density"),
        ("road_construction", "vehicles_density"),
    ],
}


# ======================================================================
#  Export helpers
# ======================================================================

def dag_to_dot(dag: Dict[str, Any], name: str = "CausalDAG") -> str:
    """Convert a DAG dict to a Graphviz DOT string.

    Args:
        dag:  A dict with ``"nodes"`` and ``"edges"`` keys.
        name: Graph name.

    Returns:
        DOT-format string that can be rendered with ``dot -Tpdf``.
    """
    lines = [f"digraph {name} {{", "  rankdir=TB;"]

    # Style mapping
    style_map = {
        "latent_confounder":    'shape=diamond, style=dashed, color=red',
        "instrumental_variable": 'shape=diamond, style=dashed, color=blue',
        "decision":             'shape=box, style=filled, fillcolor=lightyellow',
        "state":                'shape=ellipse',
        "task_param":           'shape=ellipse, style=filled, fillcolor=lightgrey',
        "observed_task_param":  'shape=ellipse, style=filled, fillcolor=lightgrey',
        "event":                'shape=octagon, color=red',
        "reward_component":     'shape=ellipse, style=filled, fillcolor=lightgreen',
        "reward":               'shape=doubleoctagon, style=filled, fillcolor=gold',
        "observation":          'shape=box, style=filled, fillcolor=lightblue',
    }

    for node in dag["nodes"]:
        ntype = node.get("type", "state")
        style = style_map.get(ntype, "shape=ellipse")
        label = node["name"]
        lines.append(f'  "{label}" [{style}];')

    for src, dst in dag["edges"]:
        lines.append(f'  "{src}" -> "{dst}";')

    lines.append("}")
    return "\n".join(lines)


def get_full_two_level_dag() -> Dict[str, Any]:
    """Return a merged two-level DAG (task + step) for documentation.

    The task-level and step-level DAGs are combined into a single graph,
    with task-level nodes feeding into step-level nodes to show the
    full causal structure.
    """
    # Combine nodes (deduplicate by name)
    seen = set()
    nodes = []
    for dag in (TASK_LEVEL_DAG, WITHIN_EPISODE_DAG):
        for node in dag["nodes"]:
            if node["name"] not in seen:
                nodes.append(node)
                seen.add(node["name"])

    # Combine edges (deduplicate)
    edges_set = set()
    for dag in (TASK_LEVEL_DAG, WITHIN_EPISODE_DAG):
        for edge in dag["edges"]:
            edges_set.add(tuple(edge))

    return {
        "description": "Full two-level causal DAG: task-level SCM + within-episode dynamics",
        "nodes": nodes,
        "edges": list(edges_set),
    }
