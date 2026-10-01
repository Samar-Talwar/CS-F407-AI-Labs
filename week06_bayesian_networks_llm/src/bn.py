# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Core Bayesian Network construction and validation for the Sprinkler network.

This module implements the trusted reference model from the Week 6 notebook.
All probabilities match the notebook specification exactly.
Native discrete Bayesian network implementation without external ML libraries.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TabularCPD:
    """
    Tabular Conditional Probability Distribution for discrete Bayesian Networks.

    Stored as a 2D float array of shape (variable_card, num_parent_configurations),
    where columns correspond to lexicographically ordered parent state combinations.
    """

    def __init__(
        self,
        variable: str,
        variable_card: int,
        values: list | np.ndarray,
        evidence: list[str] | None = None,
        evidence_card: list[int] | None = None,
        state_names: dict[str, list[Any]] | None = None,
    ) -> None:
        self.variable = variable
        self.variable_card = int(variable_card)
        self.evidence = list(evidence) if evidence else None
        self.evidence_card = list(evidence_card) if evidence_card else None
        self.state_names = state_names or {}

        val_arr = np.asarray(values, dtype=float)
        if val_arr.ndim == 1:
            val_arr = val_arr.reshape((self.variable_card, -1))
        elif val_arr.ndim == 2:
            pass
        else:
            val_arr = val_arr.reshape((self.variable_card, -1))

        self.values = val_arr

    def copy(self) -> TabularCPD:
        """Return a deep copy of this CPD."""
        return TabularCPD(
            variable=self.variable,
            variable_card=self.variable_card,
            values=self.values.copy(),
            evidence=list(self.evidence) if self.evidence else None,
            evidence_card=list(self.evidence_card) if self.evidence_card else None,
            state_names=dict(self.state_names) if self.state_names else None,
        )

    def get_values(self) -> np.ndarray:
        """Return CPD probability table."""
        return self.values

    def __repr__(self) -> str:
        return (
            f"TabularCPD(variable={self.variable!r}, "
            f"variable_card={self.variable_card}, "
            f"evidence={self.evidence!r})"
        )

    def __str__(self) -> str:
        return f"TabularCPD for {self.variable}\nValues:\n{self.values}"


class SimpleSeries:
    """Lightweight 1D series wrapper for DataFrame column indexing."""

    def __init__(self, values: np.ndarray, name: str = "") -> None:
        self.values = np.asarray(values)
        self.name = name

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, item: Any) -> Any:
        res = self.values[item]
        if isinstance(res, np.ndarray):
            return SimpleSeries(res, self.name)
        return res

    def __eq__(self, other: Any) -> np.ndarray:  # type: ignore[override]
        if isinstance(other, SimpleSeries):
            return self.values == other.values
        return self.values == other

    def __ne__(self, other: Any) -> np.ndarray:  # type: ignore[override]
        if isinstance(other, SimpleSeries):
            return self.values != other.values
        return self.values != other

    def unique(self) -> np.ndarray:
        return np.unique(self.values)

    def sum(self) -> int | float:
        return self.values.sum()

    def mean(self) -> float:
        return float(self.values.mean())

    def tolist(self) -> list:
        return self.values.tolist()

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype


class SimpleDataFrame:
    """Lightweight 2D tabular data structure replacing pandas DataFrame."""

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self._data: dict[str, np.ndarray] = {}
        if data is not None:
            if isinstance(data, dict):
                for k, v in data.items():
                    self._data[str(k)] = np.asarray(v)
            elif isinstance(data, SimpleDataFrame):
                for k, v in data._data.items():
                    self._data[k] = v.copy()
            elif isinstance(data, np.ndarray):
                cols = columns or [f"col_{i}" for i in range(data.shape[1])]
                for i, col in enumerate(cols):
                    self._data[col] = data[:, i]

        if columns is not None and not self._data:
            for col in columns:
                self._data[col] = np.array([], dtype=int)

    @property
    def columns(self) -> list[str]:
        return list(self._data.keys())

    @property
    def shape(self) -> tuple[int, int]:
        if not self._data:
            return (0, 0)
        first_col = next(iter(self._data.values()))
        return (len(first_col), len(self._data))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(f"Column {key!r} not found")
            return SimpleSeries(self._data[key], key)
        elif isinstance(key, (list, np.ndarray, SimpleSeries)):
            mask = (
                np.asarray(key, dtype=bool)
                if not isinstance(key, np.ndarray) or key.dtype != bool
                else key
            )
            if isinstance(key, SimpleSeries):
                mask = key.values.astype(bool)
            new_data = {col: arr[mask] for col, arr in self._data.items()}
            return SimpleDataFrame(new_data)
        raise TypeError(f"Invalid key type: {type(key)}")

    def equals(self, other: Any) -> bool:
        if not isinstance(other, SimpleDataFrame):
            return False
        if set(self.columns) != set(other.columns):
            return False
        if self.shape != other.shape:
            return False
        return all(
            np.array_equal(self._data[col], other._data[col])
            for col in self.columns
        )

    def to_dict(self) -> dict[str, list]:
        return {k: v.tolist() for k, v in self._data.items()}

    def copy(self) -> SimpleDataFrame:
        return SimpleDataFrame({k: v.copy() for k, v in self._data.items()})

    def __repr__(self) -> str:
        return f"SimpleDataFrame(shape={self.shape}, columns={self.columns})"


class DiscreteBayesianNetwork:
    """
    Directed Acyclic Graph with Tabular Conditional Probability Distributions.
    """

    def __init__(self, ebunch: list[tuple[str, str]] | None = None) -> None:
        self._nodes: set[str] = set()
        self._edges: set[tuple[str, str]] = set()
        self._cpds: dict[str, TabularCPD] = {}

        if ebunch is not None:
            self.add_edges_from(ebunch)

    def add_node(self, node: str) -> None:
        self._nodes.add(node)

    def add_nodes_from(self, nodes: list[str]) -> None:
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u: str, v: str) -> None:
        self.add_node(u)
        self.add_node(v)
        self._edges.add((u, v))

    def add_edges_from(self, ebunch: list[tuple[str, str]]) -> None:
        for u, v in ebunch:
            self.add_edge(u, v)

    def remove_edge(self, u: str, v: str) -> None:
        self._edges.discard((u, v))

    def nodes(self) -> list[str]:
        return sorted(self._nodes)

    def edges(self) -> list[tuple[str, str]]:
        return sorted(self._edges)

    def get_parents(self, node: str) -> list[str]:
        return sorted([u for u, v in self._edges if v == node])

    def get_children(self, node: str) -> list[str]:
        return sorted([v for u, v in self._edges if u == node])

    def add_cpds(self, *cpds: TabularCPD) -> None:
        for cpd in cpds:
            self._nodes.add(cpd.variable)
            if cpd.evidence:
                for parent in cpd.evidence:
                    self._nodes.add(parent)
            self._cpds[cpd.variable] = cpd

    def get_cpds(self, node: str | None = None) -> Any:
        if node is None:
            return list(self._cpds.values())
        if node in self._cpds:
            return self._cpds[node]
        raise ValueError(f"No CPD found for node: {node}")

    def get_cardinality(self, node: str) -> int:
        cpd = self.get_cpds(node)
        return int(cpd.variable_card)

    def remove_cpds(self, *cpds: TabularCPD | str) -> None:
        for cpd in cpds:
            var = cpd.variable if isinstance(cpd, TabularCPD) else str(cpd)
            self._cpds.pop(var, None)

    def topological_sort(self) -> list[str]:
        in_degree = {n: 0 for n in self._nodes}
        for _u, v in self._edges:
            in_degree[v] += 1

        queue = [n for n in sorted(self._nodes) if in_degree[n] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
            queue.sort()

        if len(order) != len(self._nodes):
            raise ValueError("Cycle detected in Bayesian Network structure.")
        return order

    def check_model(self) -> bool:
        """
        Validate network consistency: DAG topology, missing CPDs,
        cardinalities, evidence/parent alignment, and column normalization.
        """
        # 1. Topological check (cycles)
        self.topological_sort()

        # 2. CPD completeness
        for node in self._nodes:
            if node not in self._cpds:
                raise ValueError(f"No CPD associated with node {node}")

        # 3. CPD consistency with graph
        for node, cpd in self._cpds.items():
            if cpd.variable != node:
                raise ValueError(f"CPD variable {cpd.variable} does not match node {node}")

            parents = self.get_parents(node)
            cpd_evidence = cpd.evidence or []
            if sorted(parents) != sorted(cpd_evidence):
                raise ValueError(
                    f"Evidence {cpd_evidence} for CPD {node} does not match graph parents {parents}"
                )

            # Check normalization: column sums must equal 1
            col_sums = cpd.values.sum(axis=0)
            if not np.allclose(col_sums, 1.0, atol=1e-5):
                raise ValueError(
                    f"CPD values for {node} are not equal to 1. Column sums: {col_sums}"
                )

        return True

    def copy(self) -> DiscreteBayesianNetwork:
        new_net = DiscreteBayesianNetwork(list(self._edges))
        for node in self._nodes:
            new_net.add_node(node)
        for cpd in self._cpds.values():
            new_net.add_cpds(cpd.copy())
        return new_net

    def simulate(
        self,
        n_samples: int,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> SimpleDataFrame:
        """
        Sample observations from the joint distribution via forward topological sampling.
        Matches standard pgmpy deterministic simulation sequence given seed.
        """
        order = self.topological_sort()
        rng = np.random.default_rng(seed)
        samples: dict[str, np.ndarray] = {}

        for node in order:
            cpd = self._cpds[node]
            var_card = cpd.variable_card

            if not cpd.evidence:
                probs = cpd.values[:, 0]
                samples[node] = rng.choice(var_card, size=n_samples, p=probs)
            else:
                parents = cpd.evidence
                parent_cards = cpd.evidence_card or [2] * len(parents)
                node_samples = np.zeros(n_samples, dtype=int)

                # For each sample, compute parent configuration index
                parent_states = np.column_stack([samples[p] for p in parents])
                strides = np.cumprod([1] + parent_cards[::-1])[:-1][::-1]
                col_indices = np.sum(parent_states * strides, axis=1)

                unique_cols = np.unique(col_indices)
                for col in unique_cols:
                    mask = (col_indices == col)
                    count = int(np.sum(mask))
                    if count > 0:
                        probs = cpd.values[:, col]
                        node_samples[mask] = rng.choice(var_card, size=count, p=probs)

                samples[node] = node_samples

        return SimpleDataFrame(samples)

    def fit(self, data: Any, estimator: Any = None) -> DiscreteBayesianNetwork:
        if estimator is not None:
            cpds = estimator.get_parameters(self, data)
            self.add_cpds(*cpds)
        return self

    def query(
        self,
        variables: list[str],
        evidence: dict[str, int] | None = None,
        show_progress: bool = False,
    ) -> Any:
        from .inference import VariableElimination
        ve = VariableElimination(self)
        return ve.query(variables=variables, evidence=evidence, show_progress=show_progress)


# Alias for compatibility with pgmpy API and LLM-generated code
BayesianModel = DiscreteBayesianNetwork


def build_reference_model() -> DiscreteBayesianNetwork:
    """
    Construct the trusted Sprinkler Bayesian network.

    Graph structure:
        Cloudy -> Rain
        Cloudy -> Sprinkler
        Rain -> WetGrass
        Sprinkler -> WetGrass

    Binary states: 0 = False, 1 = True

    Probabilities (from notebook):
        P(Cloudy=1) = 0.5

        P(Rain=1 | Cloudy=0) = 0.2
        P(Rain=1 | Cloudy=1) = 0.8

        P(Sprinkler=1 | Cloudy=0) = 0.5
        P(Sprinkler=1 | Cloudy=1) = 0.1

        P(WetGrass=1 | Rain=0, Sprinkler=0) = 0.01
        P(WetGrass=1 | Rain=0, Sprinkler=1) = 0.90
        P(WetGrass=1 | Rain=1, Sprinkler=0) = 0.90
        P(WetGrass=1 | Rain=1, Sprinkler=1) = 0.99

    Returns:
        DiscreteBayesianNetwork with all CPDs added and validated.
    """
    model = DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])

    # P(Cloudy)
    cpd_cloudy = TabularCPD(
        variable="Cloudy",
        variable_card=2,
        values=[
            [0.5],  # P(Cloudy=0)
            [0.5],  # P(Cloudy=1)
        ],
    )

    # P(Rain | Cloudy)
    # Columns correspond to Cloudy=0, Cloudy=1.
    cpd_rain = TabularCPD(
        variable="Rain",
        variable_card=2,
        values=[
            [0.8, 0.2],  # P(Rain=0 | Cloudy)
            [0.2, 0.8],  # P(Rain=1 | Cloudy)
        ],
        evidence=["Cloudy"],
        evidence_card=[2],
    )

    # P(Sprinkler | Cloudy)
    # Columns correspond to Cloudy=0, Cloudy=1.
    cpd_sprinkler = TabularCPD(
        variable="Sprinkler",
        variable_card=2,
        values=[
            [0.5, 0.9],  # P(Sprinkler=0 | Cloudy)
            [0.5, 0.1],  # P(Sprinkler=1 | Cloudy)
        ],
        evidence=["Cloudy"],
        evidence_card=[2],
    )

    # P(WetGrass | Rain, Sprinkler)
    # With evidence=["Rain", "Sprinkler"], columns are ordered as:
    # (Rain=0, Sprinkler=0),
    # (Rain=0, Sprinkler=1),
    # (Rain=1, Sprinkler=0),
    # (Rain=1, Sprinkler=1).
    cpd_wetgrass = TabularCPD(
        variable="WetGrass",
        variable_card=2,
        values=[
            [0.99, 0.10, 0.10, 0.01],  # WetGrass=0
            [0.01, 0.90, 0.90, 0.99],  # WetGrass=1
        ],
        evidence=["Rain", "Sprinkler"],
        evidence_card=[2, 2],
    )

    model.add_cpds(
        cpd_cloudy,
        cpd_rain,
        cpd_sprinkler,
        cpd_wetgrass,
    )

    return model


def validate_cpt(model: DiscreteBayesianNetwork) -> bool:
    """
    Validate that all CPDs in the model are properly normalized.

    Args:
        model: DiscreteBayesianNetwork to validate.

    Returns:
        True if all CPDs sum to 1 for each parent configuration.
    """
    try:
        if not model.check_model():
            return False
    except Exception:
        return False

    for cpd in model.get_cpds():
        values = np.asarray(cpd.values)
        col_sums = values.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-5):
            return False
        if np.any(values < -1e-10):
            return False

    return True


def get_cpd_values_map(model: DiscreteBayesianNetwork) -> dict[str, np.ndarray]:
    """
    Flatten CPD arrays into a dictionary for easy numerical comparison.

    Args:
        model: DiscreteBayesianNetwork with fitted CPDs.

    Returns:
        Dictionary mapping variable name to CPD values array.
    """
    result = {}
    for cpd in model.get_cpds():
        result[cpd.variable] = np.asarray(cpd.values, dtype=float)
    return result


def compare_models(
    model_a: DiscreteBayesianNetwork, model_b: DiscreteBayesianNetwork
) -> dict[str, float]:
    """
    Compare two models by maximum absolute difference in CPD values.

    Args:
        model_a: First model.
        model_b: Second model.

    Returns:
        Dictionary mapping variable name to max absolute difference.
    """
    a = get_cpd_values_map(model_a)
    b = get_cpd_values_map(model_b)

    variables = sorted(set(a.keys()) | set(b.keys()))
    result = {}

    for variable in variables:
        if variable in a and variable in b:
            diff = np.max(np.abs(a[variable] - b[variable]))
            result[variable] = float(diff)
        else:
            result[variable] = float("inf")

    return result


def make_empty_structure() -> DiscreteBayesianNetwork:
    """
    Create the Sprinkler network structure without CPDs.

    Returns:
        DiscreteBayesianNetwork with edges but no CPDs.
    """
    return DiscreteBayesianNetwork([
        ("Cloudy", "Rain"),
        ("Cloudy", "Sprinkler"),
        ("Rain", "WetGrass"),
        ("Sprinkler", "WetGrass"),
    ])
