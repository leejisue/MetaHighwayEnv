"""Declarative Structural Causal Model (SCM) graph engine.

This module provides a general-purpose :class:`SCMGraph` that replaces
the hardcoded linear SCM in :class:`CausalTaskDistribution` with a
*declarative* graph specification.  Users can define arbitrary DAGs
with multiple latent confounders, observable variables, and non-linear
structural equations.

Design goals
~~~~~~~~~~~~
- **Declarative**: define the graph once, then ``sample()`` repeatedly.
- **Extensible**: add confounders, observables, or edges at any time
  before the first ``sample()`` call.
- **Interventional**: ``sample(interventions={"X": 5.0})`` breaks the
  incoming edges to ``X`` and fixes its value (do-calculus).
- **Exportable**: ``to_dot()`` produces a Graphviz DOT string for
  visualisation.

Example::

    from highway_env.meta_rl.scm import SCMGraph

    scm = SCMGraph(seed=42)

    # Latent confounders
    scm.add_node("weather", node_type="latent", low=0.0, high=1.0)
    scm.add_node("road_quality", node_type="latent", low=0.0, high=1.0)

    # Observables
    scm.add_node("speed_limit", node_type="observed", low=15.0, high=30.0)
    scm.add_node("driver_aggressiveness", node_type="observed", low=0.0, high=1.0)
    scm.add_node("vehicles_density", node_type="observed", low=0.5, high=2.0)

    # Causal edges  (linear by default)
    scm.add_edge("weather", "speed_limit", weight=-15.0, intercept=30.0)
    scm.add_edge("weather", "driver_aggressiveness", weight=0.8, intercept=0.1)
    scm.add_edge("weather", "vehicles_density", weight=-1.5, intercept=2.0)
    scm.add_edge("road_quality", "driver_aggressiveness", weight=-0.3)

    # Sample observational data
    sample = scm.sample()
    # {'weather': 0.73, 'road_quality': 0.41, 'speed_limit': 19.1, ...}

    # Sample interventional data: do(speed_limit := 25)
    sample_do = scm.sample(interventions={"speed_limit": 25.0})
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  Built-in mechanism functions
# ---------------------------------------------------------------------------

def _linear_mechanism(parent_value: float, weight: float, intercept: float) -> float:
    """Default linear mechanism: ``intercept + weight * parent_value``."""
    return intercept + weight * parent_value


def _sigmoid_mechanism(parent_value: float, weight: float, intercept: float) -> float:
    """Sigmoid (logistic) mechanism for saturating nonlinear relationships.

    The output is::

        intercept + weight * σ(steepness * (parent_value − midpoint))

    where ``σ`` is the standard logistic function, ``steepness = 10``, and
    ``midpoint = 0.5``.  The steepness and midpoint are baked in so that
    the mechanism has the same ``(parent_value, weight, intercept)``
    signature as all other mechanisms.

    Behaviour:
        - parent ≈ 0  →  value ≈ intercept + weight * σ(-5) ≈ intercept
        - parent ≈ 0.5 →  value ≈ intercept + weight * 0.5
        - parent ≈ 1  →  value ≈ intercept + weight * σ(+5) ≈ intercept + weight

    The transition is sharper than a linear mapping, creating a clear
    S-shaped nonlinearity that is easy to visualise in paper figures.
    """
    steepness = 10.0
    midpoint = 0.5
    sigmoid = 1.0 / (1.0 + np.exp(-steepness * (parent_value - midpoint)))
    return intercept + weight * sigmoid


def _quadratic_mechanism(parent_value: float, weight: float, intercept: float) -> float:
    """Quadratic (polynomial) mechanism: ``intercept + weight * parent_value²``.

    Produces a convex (weight > 0) or concave (weight < 0) response.
    Combined with a linear edge this can model interactions where the
    marginal effect of the parent *increases* with its magnitude.
    """
    return intercept + weight * (parent_value ** 2)


def _threshold_mechanism(parent_value: float, weight: float, intercept: float) -> float:
    """Threshold (step) mechanism for regime-switching relationships.

    The output is::

        intercept + weight * 1[parent_value >= 0.5]

    i.e. the child variable jumps discretely when the parent reaches
    the threshold.  This models regime changes like "fog visibility
    drops sharply when weather severity reaches 0.5".  The boundary
    point ``parent_value == 0.5`` is treated as already in the harsh
    regime (>=) so that discrete confounder samples at exactly 0.5
    fall on the post-threshold side, matching the docstring of
    :meth:`SCMGraph.default_highway_scm_nonlinear` (Fix E).
    """
    return intercept + weight * float(parent_value >= 0.5)


# Registry of named mechanism functions for describe() / to_dot()
_MECHANISM_REGISTRY: Dict[Callable, str] = {
    _linear_mechanism: "linear",
    _sigmoid_mechanism: "sigmoid",
    _quadratic_mechanism: "quadratic",
    _threshold_mechanism: "threshold",
}


class SCMNode:
    """A node (variable) in the SCM graph."""

    __slots__ = ("name", "node_type", "low", "high", "distribution")

    def __init__(
        self,
        name: str,
        node_type: str = "observed",
        low: float = 0.0,
        high: float = 1.0,
        distribution: str = "uniform",
    ):
        assert node_type in ("latent", "observed"), (
            f"node_type must be 'latent' or 'observed', got {node_type!r}"
        )
        self.name = name
        self.node_type = node_type
        self.low = low
        self.high = high
        self.distribution = distribution  # "uniform" or "discrete_3"


class SCMEdge:
    """A directed causal edge in the SCM graph."""

    __slots__ = ("parent", "child", "weight", "intercept", "mechanism")

    def __init__(
        self,
        parent: str,
        child: str,
        weight: float = 1.0,
        intercept: float = 0.0,
        mechanism: Optional[Callable[[float, float, float], float]] = None,
    ):
        self.parent = parent
        self.child = child
        self.weight = weight
        self.intercept = intercept
        self.mechanism = mechanism or _linear_mechanism


class SCMGraph:
    """Declarative Structural Causal Model as a directed acyclic graph.

    Nodes represent causal variables (latent or observed).
    Edges represent structural equations: ``child = f(parent) + ε``.

    When a child has *multiple* parents, the contributions are summed::

        child = Σ_i f_i(parent_i) + ε

    For root nodes (no parents), the value is sampled from the node's
    prior distribution and clipped to ``[low, high]``.

    Parameters
    ----------
    noise_scale : float
        Relative noise standard deviation (fraction of variable range).
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        noise_scale: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self._nodes: Dict[str, SCMNode] = {}
        self._edges: List[SCMEdge] = []
        self._topo_order: Optional[List[str]] = None  # cached

    # ------------------------------------------------------------------
    #  Graph construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        node_type: str = "observed",
        low: float = 0.0,
        high: float = 1.0,
        distribution: str = "uniform",
    ) -> SCMGraph:
        """Add a variable (node) to the SCM.

        Args:
            name:         Variable name.
            node_type:    ``"latent"`` or ``"observed"``.
            low, high:    Clipping range for sampled values.
            distribution: ``"uniform"`` (default) or ``"discrete_3"``
                          (samples from {low, mid, high}).

        Returns:
            self (for chaining).
        """
        self._nodes[name] = SCMNode(name, node_type, low, high, distribution)
        self._topo_order = None  # invalidate cache
        return self

    def add_edge(
        self,
        parent: str,
        child: str,
        weight: float = 1.0,
        intercept: float = 0.0,
        mechanism: Optional[Callable[[float, float, float], float]] = None,
    ) -> SCMGraph:
        """Add a causal edge parent → child.

        The structural equation for this edge is::

            contribution = mechanism(parent_value, weight, intercept)

        Default mechanism: ``intercept + weight * parent_value``.

        When a child has multiple parents, contributions are summed.

        Args:
            parent:    Source node name.
            child:     Target node name.
            weight:    Coefficient.
            intercept: Bias term (only the *first* edge to a child should
                       set this; subsequent edges can leave it at 0).
            mechanism: Custom ``(parent_val, weight, intercept) -> float``.

        Returns:
            self (for chaining).
        """
        if parent not in self._nodes:
            raise ValueError(f"Parent node {parent!r} not in graph")
        if child not in self._nodes:
            raise ValueError(f"Child node {child!r} not in graph")
        self._edges.append(
            SCMEdge(parent, child, weight, intercept, mechanism)
        )
        self._topo_order = None
        return self

    # ------------------------------------------------------------------
    #  Topological sort
    # ------------------------------------------------------------------

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological ordering."""
        if self._topo_order is not None:
            return self._topo_order

        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        children_of: Dict[str, List[str]] = {n: [] for n in self._nodes}
        for e in self._edges:
            in_degree[e.child] += 1
            children_of[e.parent].append(e.child)

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children_of[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._nodes):
            raise ValueError("SCM graph contains a cycle!")

        self._topo_order = order
        return order

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        interventions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Sample one realisation from the SCM.

        Args:
            interventions: ``{variable: value}`` for do-operator.
                           Intervened variables ignore their parents
                           and take the fixed value.

        Returns:
            Dict mapping every node name to its sampled value.
        """
        interventions = interventions or {}
        order = self._topological_sort()
        values: Dict[str, float] = {}

        for name in order:
            node = self._nodes[name]

            # Intervention: fix value, ignore parents
            if name in interventions:
                values[name] = float(interventions[name])
                continue

            # Collect parent contributions
            parent_edges = [e for e in self._edges if e.child == name]

            if not parent_edges:
                # Root node: sample from prior
                values[name] = self._sample_root(node)
            else:
                # Child node: sum parent contributions + noise
                total = 0.0
                for edge in parent_edges:
                    parent_val = values[edge.parent]
                    total += edge.mechanism(parent_val, edge.weight, edge.intercept)

                noise = self.rng.normal(
                    0.0, self.noise_scale * (node.high - node.low)
                )
                values[name] = float(
                    np.clip(total + noise, node.low, node.high)
                )

        return values

    def sample_batch(
        self,
        n: int,
        interventions: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, float]]:
        """Sample ``n`` realisations."""
        return [self.sample(interventions) for _ in range(n)]

    def _sample_root(self, node: SCMNode) -> float:
        """Sample a root node from its prior distribution."""
        if node.distribution == "discrete_3":
            mid = (node.low + node.high) / 2.0
            return float(self.rng.choice([node.low, mid, node.high]))
        else:  # "uniform"
            return float(self.rng.uniform(node.low, node.high))

    # ------------------------------------------------------------------
    #  Introspection
    # ------------------------------------------------------------------

    def get_parents(self, node: str) -> List[str]:
        """Return parent names of a node."""
        return [e.parent for e in self._edges if e.child == node]

    def get_children(self, node: str) -> List[str]:
        """Return child names of a node."""
        return [e.child for e in self._edges if e.parent == node]

    def latent_nodes(self) -> List[str]:
        """Return names of latent (hidden) nodes."""
        return [n for n, nd in self._nodes.items() if nd.node_type == "latent"]

    def observed_nodes(self) -> List[str]:
        """Return names of observed nodes."""
        return [n for n, nd in self._nodes.items() if nd.node_type == "observed"]

    def describe(self) -> str:
        """Human-readable summary of the SCM.

        Non-linear edges are annotated with their mechanism name
        (e.g. ``sigmoid``, ``quadratic``) so the reader can
        immediately see which relationships are non-linear.
        """
        lines = ["SCM Graph:"]
        for name in self._topological_sort():
            node = self._nodes[name]
            parents = self.get_parents(name)
            tag = "[latent]" if node.node_type == "latent" else "[observed]"
            if parents:
                edges_desc = []
                for e in self._edges:
                    if e.child == name:
                        mech_name = _MECHANISM_REGISTRY.get(e.mechanism, "custom")
                        sign = "+" if e.weight >= 0 else ""
                        if mech_name == "linear":
                            edges_desc.append(
                                f"{e.intercept}{sign}{e.weight}*{e.parent}"
                            )
                        else:
                            edges_desc.append(
                                f"{mech_name}({e.parent}, w={e.weight}, b={e.intercept})"
                            )
                eq = " + ".join(edges_desc) + " + ε"
                lines.append(
                    f"  {name} {tag} = {eq}  ∈ [{node.low}, {node.high}]"
                )
            else:
                lines.append(
                    f"  {name} {tag} ~ {node.distribution}"
                    f"[{node.low}, {node.high}]  (root)"
                )
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Export as Graphviz DOT string for visualisation.

        Non-linear edges include the mechanism name in their label
        (e.g. ``sigmoid, w=−15.0``) so the graph is self-documenting.
        """
        lines = ["digraph SCM {", "  rankdir=TB;"]
        for name, node in self._nodes.items():
            shape = "ellipse" if node.node_type == "observed" else "diamond"
            style = "" if node.node_type == "observed" else ', style="dashed"'
            lines.append(f'  "{name}" [shape={shape}{style}];')
        for edge in self._edges:
            mech_name = _MECHANISM_REGISTRY.get(edge.mechanism, "custom")
            if mech_name == "linear":
                label = f"{edge.weight:+.2f}"
            else:
                label = f"{mech_name}, w={edge.weight:+.2f}"
            lines.append(
                f'  "{edge.parent}" -> "{edge.child}" '
                f'[label="{label}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Ground truth adjacency matrix (for SHD evaluation)
    # ------------------------------------------------------------------

    def get_adjacency_matrix(
        self,
        variable_order: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Return the ground truth binary adjacency matrix.

        A[i, j] = 1 iff there is a directed edge from variable j to variable i
        (i.e. j is a parent of i).  This convention matches CUTS+ / NOTEARS.

        Args:
            variable_order: Explicit ordering of variables.  If ``None``,
                uses topological order.

        Returns:
            (A, names): binary adjacency matrix and list of variable names.
        """
        if variable_order is None:
            variable_order = self._topological_sort()
        idx = {name: i for i, name in enumerate(variable_order)}
        n = len(variable_order)
        A = np.zeros((n, n), dtype=np.float64)
        for edge in self._edges:
            if edge.parent in idx and edge.child in idx:
                # A[child, parent] = 1 (parent → child)
                A[idx[edge.child], idx[edge.parent]] = 1.0
        return A, variable_order

    def get_observed_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return adjacency matrix over observed variables only.

        Useful for comparing with CausalComRL's learned DAG A matrix
        which only models observed causal concepts (z1_dim = num observed vars).

        Returns:
            (A, names): binary adjacency matrix and list of observed variable names.
        """
        obs_vars = self.observed_nodes()
        return self.get_adjacency_matrix(variable_order=obs_vars)

    # ------------------------------------------------------------------
    #  Factory: recreate the default weather SCM
    # ------------------------------------------------------------------

    @classmethod
    def default_highway_scm(
        cls,
        noise_scale: float = 0.05,
        seed: Optional[int] = None,
    ) -> SCMGraph:
        """Create the default highway SCM matching CausalTaskDistribution.

        Graph::

            weather (latent)  ──→  speed_limit
                              ──→  driver_aggressiveness
                              ──→  vehicles_density
        """
        scm = cls(noise_scale=noise_scale, seed=seed)
        scm.add_node("weather", "latent", 0.0, 1.0)
        scm.add_node("speed_limit", "observed", 15.0, 30.0)
        scm.add_node("driver_aggressiveness", "observed", 0.0, 1.0)
        scm.add_node("vehicles_density", "observed", 0.5, 2.0)

        scm.add_edge("weather", "speed_limit", weight=-15.0, intercept=30.0)
        scm.add_edge("weather", "driver_aggressiveness", weight=0.8, intercept=0.1)
        scm.add_edge("weather", "vehicles_density", weight=-1.5, intercept=2.0)
        return scm

    @classmethod
    def default_highway_scm_with_iv(
        cls,
        noise_scale: float = 0.05,
        seed: Optional[int] = None,
    ) -> SCMGraph:
        """Create the highway SCM with an instrumental variable.

        Extends the default SCM with ``road_construction`` as an
        instrumental variable (IV) for ``vehicles_density``:

        - ``road_construction`` is sampled independently of ``weather``
        - ``road_construction → vehicles_density`` (causal effect)
        - ``road_construction`` has NO direct effect on reward or
          ``driver_aggressiveness``

        This satisfies the IV conditions::

            Z = road_construction
            X = vehicles_density  (treatment)
            Y = reward            (outcome)
            U = weather           (confounder)

            Z ⊥ U              (independence)
            Z → X              (relevance)
            Z ↛ Y | X          (exclusion restriction — no direct effect)

        Graph::

            weather (latent)  ──→  speed_limit
                              ──→  driver_aggressiveness
                              ──→  vehicles_density  ←── road_construction (IV)
        """
        scm = cls.default_highway_scm(noise_scale=noise_scale, seed=seed)

        # Add the instrumental variable (independent of weather)
        scm.add_node(
            "road_construction",
            node_type="observed",
            low=0.0,
            high=1.0,
            distribution="uniform",
        )

        # IV → treatment (vehicles_density)
        # road_construction pushes density down (construction = fewer lanes)
        # Fix D: weight reduced from -0.8 to -0.3 so the combined
        # density signal (2.0 - 1.5*weather - 0.3*road_construction)
        # stays inside the [0.5, 2.0] clip range for the vast majority
        # of (weather, road_construction) combinations.  The IV
        # exclusion-restriction is unaffected; only the IV strength
        # changes.
        scm.add_edge(
            "road_construction", "vehicles_density",
            weight=-0.3,
            intercept=0.0,  # additive to weather's contribution
        )

        return scm

    # ------------------------------------------------------------------
    #  Factory: nonlinear highway SCM
    # ------------------------------------------------------------------

    @classmethod
    def default_highway_scm_nonlinear(
        cls,
        noise_scale: float = 0.05,
        seed: Optional[int] = None,
    ) -> SCMGraph:
        """Create a **nonlinear** highway SCM.

        Same graph topology as :meth:`default_highway_scm` (weather →
        speed_limit, driver_aggressiveness, vehicles_density), but every
        edge uses a distinct nonlinear mechanism so that the task-level
        causal relationships are clearly nonlinear:

        - **weather → speed_limit**: *sigmoid* — speed limit drops
          sharply once weather severity crosses ≈ 0.5, modelling a
          regime switch (e.g. fog threshold triggers a speed advisory).
        - **weather → driver_aggressiveness**: *threshold* — drivers
          behave calmly in good weather and switch to aggressive
          driving beyond severity 0.5 (panic / rush to exit highway).
        - **weather → vehicles_density**: *quadratic* — density
          decreases slowly in mild weather but drops rapidly in
          severe weather (drivers stay home when conditions worsen).

        These nonlinearities are designed to be:
        1. *Interpretable* — each has a plausible traffic-domain story.
        2. *Visually distinct* — an aggressiveness-vs-weather scatter
           plot from observational data will NOT lie on a straight line.
        3. *Identifiable* — interventional and counterfactual queries
           via :meth:`sample` still work correctly.

        Graph::

            weather (latent)  ──sigmoid──→  speed_limit
                              ──threshold──→  driver_aggressiveness
                              ──quadratic──→  vehicles_density

        Recommended for paper experiments:
            Use ``default_highway_scm_nonlinear()`` (or its IV variant)
            for all main results to demonstrate that C2A handles
            nonlinear causal relationships.  The linear
            :meth:`default_highway_scm` can serve as an ablation.

        Example::

            scm_nl = SCMGraph.default_highway_scm_nonlinear(seed=42)
            samples = scm_nl.sample_batch(500)
            # Plot speed_limit vs weather: sigmoid S-curve, NOT a line

        See Also:
            :meth:`default_highway_scm` — linear version for ablation.
            :meth:`default_highway_scm_nonlinear_with_iv` — this SCM
            plus the ``road_construction`` instrumental variable.
        """
        scm = cls(noise_scale=noise_scale, seed=seed)

        # Latent confounder
        scm.add_node("weather", "latent", 0.0, 1.0)

        # Observables (same ranges as linear SCM)
        scm.add_node("speed_limit", "observed", 15.0, 30.0)
        scm.add_node("driver_aggressiveness", "observed", 0.0, 1.0)
        scm.add_node("vehicles_density", "observed", 0.5, 2.0)

        # ── Sigmoid: weather → speed_limit ──
        # At weather ≈ 0: speed_limit ≈ 30 + (-15)*σ(-5) ≈ 30
        # At weather ≈ 1: speed_limit ≈ 30 + (-15)*σ(+5) ≈ 15
        # Transition is S-shaped around weather = 0.5
        scm.add_edge(
            "weather", "speed_limit",
            weight=-15.0, intercept=30.0,
            mechanism=_sigmoid_mechanism,
        )

        # ── Threshold: weather → driver_aggressiveness ──
        # At weather <  0.5: aggressiveness ≈ 0.1
        # At weather >= 0.5: aggressiveness ≈ 0.1 + 0.8 = 0.9
        # Sharp regime switch (boundary 0.5 is in harsh regime; Fix E)
        scm.add_edge(
            "weather", "driver_aggressiveness",
            weight=0.8, intercept=0.1,
            mechanism=_threshold_mechanism,
        )

        # ── Quadratic: weather → vehicles_density ──
        # density = 2.0 + (-1.5) * weather²
        # At weather = 0: density ≈ 2.0
        # At weather = 0.5: density ≈ 2.0 - 1.5*0.25 = 1.625
        # At weather = 1: density ≈ 2.0 - 1.5 = 0.5
        # Drops slowly at first, then accelerates
        scm.add_edge(
            "weather", "vehicles_density",
            weight=-1.5, intercept=2.0,
            mechanism=_quadratic_mechanism,
        )

        return scm

    @classmethod
    def default_highway_scm_nonlinear_with_iv(
        cls,
        noise_scale: float = 0.05,
        seed: Optional[int] = None,
    ) -> SCMGraph:
        """Nonlinear highway SCM with instrumental variable.

        Combines :meth:`default_highway_scm_nonlinear` with the
        ``road_construction`` IV from :meth:`default_highway_scm_with_iv`.

        The IV edge (``road_construction → vehicles_density``) remains
        **linear** because the IV's causal effect on density is additive
        and does not depend on weather.  Only the weather edges are
        nonlinear.

        Graph::

            weather (latent)  ──sigmoid──→    speed_limit
                              ──threshold──→  driver_aggressiveness
                              ──quadratic──→  vehicles_density  ←── road_construction (IV, linear)

        See Also:
            :meth:`default_highway_scm_nonlinear` — without IV.
            :meth:`default_highway_scm_with_iv` — linear version with IV.
        """
        scm = cls.default_highway_scm_nonlinear(
            noise_scale=noise_scale, seed=seed,
        )

        # Add IV (same as linear version — IV effect is linear/additive)
        scm.add_node(
            "road_construction",
            node_type="observed",
            low=0.0,
            high=1.0,
            distribution="uniform",
        )
        # Fix D: weight reduced from -0.8 to -0.3 (see linear-IV factory).
        scm.add_edge(
            "road_construction", "vehicles_density",
            weight=-0.3,
            intercept=0.0,
        )

        return scm


# =====================================================================
#  Standalone utility: ground truth adjacency for a given scm_type
# =====================================================================

def get_true_adjacency_for_scm_type(
    scm_type: str,
    observed_only: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """Return the ground truth adjacency matrix for a named SCM variant.

    Args:
        scm_type: One of ``"linear"``, ``"nonlinear"``, ``"linear_iv"``,
                  ``"nonlinear_iv"``.
        observed_only: If ``True``, return only observed variables
                       (excludes latent confounders like ``weather``).
        seed: Random seed (only affects noise, not structure).

    Returns:
        (A, variable_names): binary adjacency matrix ``A[child, parent]``
        and the corresponding list of variable names.
    """
    _FACTORIES = {
        "linear":       SCMGraph.default_highway_scm,
        "nonlinear":    SCMGraph.default_highway_scm_nonlinear,
        "linear_iv":    SCMGraph.default_highway_scm_with_iv,
        "nonlinear_iv": SCMGraph.default_highway_scm_nonlinear_with_iv,
    }
    if scm_type not in _FACTORIES:
        raise ValueError(
            f"Unknown scm_type {scm_type!r}. "
            f"Choose from {list(_FACTORIES.keys())}"
        )
    scm = _FACTORIES[scm_type](seed=seed)
    if observed_only:
        return scm.get_observed_adjacency_matrix()
    return scm.get_adjacency_matrix()


def count_accuracy(W_true: np.ndarray, W_est: np.ndarray) -> Dict[str, float]:
    """Compute SHD, TPR, FPR, FDR, F1 between true and estimated graphs.

    Standalone reimplementation (no external dependency) of the metric
    from ``analyze_utils.py`` with an added F1 score.

    Args:
        W_true: Ground truth adjacency matrix (binary or weighted).
        W_est:  Estimated adjacency matrix (binary or weighted).

    Returns:
        Dict with keys: ``shd``, ``tpr``, ``fpr``, ``fdr``, ``f1``,
        ``pred_size``.
    """
    B_true = (W_true != 0).astype(int)
    B_est = (W_est != 0).astype(int)
    d = B_true.shape[0]

    pred = np.flatnonzero(B_est)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)

    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    pred_size = len(pred)
    # Fix F: NOTEARS convention for the false-positive denominator.
    # ``cond_neg_size`` is the count of *non-edges* in the undirected
    # skeleton sense: 0.5*d*(d-1) is the number of unordered node pairs
    # (skeleton-edge slots), minus the number of true directed edges.
    # This mixes directed (len(cond)) and undirected (skeleton) counts
    # but matches the standard NOTEARS metric implementation.
    # Reference: https://github.com/xunzheng/notears/blob/master/notears/utils.py
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)

    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    # SHD: undirected extra + undirected missing + reverse
    B_lower = np.tril(B_est + B_est.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # F1 score
    precision = 1.0 - fdr
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-8)

    return {
        "shd": shd,
        "tpr": tpr,
        "fpr": fpr,
        "fdr": fdr,
        "f1": f1,
        "pred_size": pred_size,
    }
