# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Exact inference for discrete Bayesian Networks.

Implements exact Variable Elimination and brute-force Joint Distribution Enumeration
as an independent verification oracle.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .bn import DiscreteBayesianNetwork, build_reference_model


class DiscreteFactor:
    """
    Multi-dimensional discrete factor for exact variable elimination.
    """

    def __init__(
        self,
        variables: list[str],
        cardinality: list[int],
        values: np.ndarray | list,
    ) -> None:
        self.variables = list(variables)
        self.cardinality = [int(c) for c in cardinality]
        val_arr = np.asarray(values, dtype=float)
        if val_arr.shape != tuple(self.cardinality):
            val_arr = val_arr.reshape(tuple(self.cardinality))
        self.values = val_arr

    def copy(self) -> DiscreteFactor:
        return DiscreteFactor(
            variables=list(self.variables),
            cardinality=list(self.cardinality),
            values=self.values.copy(),
        )

    def reduce(self, evidence: list[tuple[str, int]]) -> DiscreteFactor:
        """
        Reduce factor by instantiating observed evidence variables.
        """
        phi = self.copy()
        for var, state in evidence:
            if var in phi.variables:
                axis = phi.variables.index(var)
                # Select the slice for variable == state
                slicer = [slice(None)] * len(phi.variables)
                slicer[axis] = state
                new_values = phi.values[tuple(slicer)]
                new_vars = [v for i, v in enumerate(phi.variables) if i != axis]
                new_card = [c for i, c in enumerate(phi.cardinality) if i != axis]
                phi = DiscreteFactor(new_vars, new_card, new_values)
        return phi

    def product(self, other: DiscreteFactor) -> DiscreteFactor:
        """
        Compute factor product phi1 * phi2.
        Align axes by expanding both factors to the union variable set,
        using the variable order of self first, then new variables from other.
        """
        # Union variables: self's order first, then other's new vars
        all_vars = list(self.variables)
        for v in other.variables:
            if v not in all_vars:
                all_vars.append(v)

        card_map = {}
        for v, c in zip(self.variables, self.cardinality, strict=False):
            card_map[v] = c
        for v, c in zip(other.variables, other.cardinality, strict=False):
            card_map[v] = c

        target_card = [card_map[v] for v in all_vars]

        def expand(values, factor_vars, target_vars, card_map):
            val = np.asarray(values, dtype=float)
            current_vars = list(factor_vars)
            # Add missing variables as trailing axes of size 1
            for v in target_vars:
                if v not in current_vars:
                    val = np.expand_dims(val, axis=-1)
                    current_vars.append(v)
            # Now current_vars matches target_vars order (with missing appended)
            # Transpose to exact target_vars order
            perm = [current_vars.index(v) for v in target_vars]
            val = np.transpose(val, perm)
            return val

        v1 = expand(self.values, self.variables, all_vars, card_map)
        v2 = expand(other.values, other.variables, all_vars, card_map)

        new_vals = v1 * v2
        return DiscreteFactor(all_vars, target_card, new_vals)

    def marginalize(self, variables: list[str]) -> DiscreteFactor:
        """
        Marginalize (sum out) given variables from this factor.
        """
        axes_to_sum = [self.variables.index(v) for v in variables if v in self.variables]
        if not axes_to_sum:
            return self.copy()

        new_values = np.sum(self.values, axis=tuple(axes_to_sum))
        new_vars = [v for i, v in enumerate(self.variables) if i not in axes_to_sum]
        new_card = [c for i, c in enumerate(self.cardinality) if i not in axes_to_sum]
        return DiscreteFactor(new_vars, new_card, new_values)

    def normalize(self, inplace: bool = True) -> DiscreteFactor:
        """
        Normalize factor values so they sum to 1.0.
        """
        total = np.sum(self.values)
        if total > 0:
            if inplace:
                self.values = self.values / total
                return self
            return DiscreteFactor(self.variables, self.cardinality, self.values / total)
        return self

    def __repr__(self) -> str:
        return f"DiscreteFactor(variables={self.variables}, cardinality={self.cardinality})"


class VariableElimination:
    """
    Exact inference engine via Variable Elimination algorithm.
    """

    def __init__(self, model: DiscreteBayesianNetwork) -> None:
        self.model = model

    def _cpd_to_factor(self, cpd: Any) -> DiscreteFactor:
        """Convert a TabularCPD to a DiscreteFactor."""
        if not cpd.evidence:
            return DiscreteFactor(
                variables=[cpd.variable],
                cardinality=[cpd.variable_card],
                values=cpd.values.flatten(),
            )

        parents = cpd.evidence
        parent_cards = cpd.evidence_card or [2] * len(parents)

        # In TabularCPD, columns are lexicographically ordered parent states:
        # col = sum(parent_state * stride)
        # We reshape values to (parent_card_1, parent_card_2, ..., var_card)
        # and then transpose to (var, parent_1, parent_2, ...)
        shape = tuple(parent_cards) + (cpd.variable_card,)
        reshaped = cpd.values.T.reshape(shape)

        # Transpose so cpd.variable is first axis
        perm = (len(parents),) + tuple(range(len(parents)))
        final_values = np.transpose(reshaped, perm)

        vars_list = [cpd.variable] + parents
        card_list = [cpd.variable_card] + parent_cards
        return DiscreteFactor(vars_list, card_list, final_values)

    def query(
        self,
        variables: list[str],
        evidence: dict[str, int] | None = None,
        elimination_order: list[str] | None = None,
        show_progress: bool = False,
    ) -> DiscreteFactor:
        """
        Compute marginal posterior distribution P(variables | evidence).
        """
        evidence = evidence or {}
        all_nodes = set(self.model.nodes())

        # Validate variables and evidence
        unknown_vars = set(variables) - all_nodes
        if unknown_vars:
            raise ValueError(f"Unknown variable(s) in query: {unknown_vars}")
        unknown_ev = set(evidence.keys()) - all_nodes
        if unknown_ev:
            raise ValueError(f"Unknown variable(s) in evidence: {unknown_ev}")

        # Check evidence states
        for k, v in evidence.items():
            cpd = self.model.get_cpds(k)
            card = cpd.variable_card if cpd is not None else 2
            if v < 0 or v >= card:
                raise ValueError(f"Invalid state {v} for variable {k} with cardinality {card}")

        # 1. Convert all CPDs to factors
        factors = [self._cpd_to_factor(cpd) for cpd in self.model.get_cpds()]

        # Separate query variables that also appear in evidence
        query_vars = list(variables)
        evidence_vars = list(evidence.keys())
        non_query_evidence = {k: v for k, v in evidence.items() if k not in query_vars}
        query_evidence = {k: v for k, v in evidence.items() if k in query_vars}

        # 2. Reduce factors with non-query evidence
        factors = [f.reduce([(k, int(v)) for k, v in non_query_evidence.items()]) for f in factors]

        # 3. Determine elimination variables (exclude query vars and evidence vars from elimination)
        query_set = set(query_vars)
        evidence_set = set(evidence_vars)
        to_eliminate = all_nodes - query_set - evidence_set

        if elimination_order is not None:
            elim_list = [v for v in elimination_order if v in to_eliminate]
        else:
            # Simple min-degree elimination order
            elim_list = sorted(to_eliminate)

        # 4. Eliminate variables
        for var in elim_list:
            var_factors = [f for f in factors if var in f.variables]
            other_factors = [f for f in factors if var not in f.variables]

            if not var_factors:
                continue

            product_factor = var_factors[0]
            for f in var_factors[1:]:
                product_factor = product_factor.product(f)

            marginalized = product_factor.marginalize([var])
            factors = other_factors + [marginalized]

        # 5. Multiply remaining factors
        if not factors:
            raise RuntimeError("No factors remaining after elimination.")

        final_product = factors[0]
        for f in factors[1:]:
            final_product = final_product.product(f)

        # 6. Apply query evidence constraint if any query variable was observed
        if query_evidence:
            for q_var, q_val in query_evidence.items():
                if q_var in final_product.variables:
                    axis = final_product.variables.index(q_var)
                    slicer = [slice(None)] * len(final_product.variables)
                    for state in range(final_product.cardinality[axis]):
                        if state != q_val:
                            slicer[axis] = state
                            final_product.values[tuple(slicer)] = 0.0

        # 7. Align variable order if needed
        if (
            final_product.variables != query_vars
            and set(final_product.variables) == set(query_vars)
        ):
            perm = [final_product.variables.index(v) for v in query_vars]
            final_product.values = np.transpose(final_product.values, perm)
            final_product.cardinality = [final_product.cardinality[p] for p in perm]
            final_product.variables = list(query_vars)

        # 8. Normalize
        final_product.normalize(inplace=True)
        return final_product


def query_posterior(
    model: DiscreteBayesianNetwork,
    variables: list[str],
    evidence: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Query posterior distribution for a single or multiple variables.

    Returns:
        1D numpy array of probabilities for single variable queries,
        or multi-dim array for joint queries.
    """
    ve = VariableElimination(model)
    factor = ve.query(variables=variables, evidence=evidence)
    return np.asarray(factor.values).flatten()


def enumerate_joint(model: DiscreteBayesianNetwork) -> dict[tuple[tuple[str, int], ...], float]:
    """
    Compute full joint distribution by exact topological factorization.

    P(C, R, S, W) = P(C) * P(R | C) * P(S | C) * P(W | R, S)

    Returns:
        Dictionary mapping complete assignments to exact joint probability.
    """
    nodes = model.nodes()
    joint: dict[tuple[tuple[str, int], ...], float] = {}

    for states in itertools.product([0, 1], repeat=len(nodes)):
        assignment = dict(zip(nodes, states, strict=False))
        prob = 1.0

        for cpd in model.get_cpds():
            var = cpd.variable
            var_state = assignment[var]

            if not cpd.evidence:
                prob *= cpd.values[var_state, 0]
            else:
                parents = cpd.evidence
                parent_cards = cpd.evidence_card or [2] * len(parents)
                col_idx = 0
                for parent, card in zip(parents, parent_cards, strict=False):
                    col_idx = col_idx * card + assignment[parent]
                prob *= cpd.values[var_state, col_idx]

        key = tuple(sorted(assignment.items()))
        joint[key] = prob

    return joint


def compute_posterior_by_enumeration(
    model: DiscreteBayesianNetwork,
    query_var: str,
    query_state: int,
    evidence: dict[str, int],
) -> float:
    """
    Independent oracle: compute P(query_var=query_state | evidence)
    by summing over all joint distribution assignments.
    """
    joint = enumerate_joint(model)
    matching_evidence_sum = 0.0
    matching_both_sum = 0.0

    for assignment_tuple, prob in joint.items():
        assignment = dict(assignment_tuple)

        # Check evidence match
        evidence_matches = all(assignment.get(k) == v for k, v in evidence.items())
        if evidence_matches:
            matching_evidence_sum += prob
            if assignment.get(query_var) == query_state:
                matching_both_sum += prob

    if matching_evidence_sum == 0.0:
        return 0.0

    return matching_both_sum / matching_evidence_sum


def compute_posterior_by_enumeration_dict(
    model: DiscreteBayesianNetwork,
    variables: list[str],
    evidence: dict[str, int],
) -> dict[int, float]:
    """
    Compute posterior distribution mapping states {0: P0, 1: P1} for a single variable.
    """
    query_var = variables[0]
    p0 = compute_posterior_by_enumeration(model, query_var, 0, evidence)
    p1 = compute_posterior_by_enumeration(model, query_var, 1, evidence)
    return {0: p0, 1: p1}


def run_inference_exercises() -> dict[str, Any]:
    """
    Run all inference exercises specified in the Week 6 notebook.

    Returns:
        Dictionary of exact posterior probabilities and oracle verification results.
    """
    model = build_reference_model()

    # Query 1: P(Rain=1 | WetGrass=1)
    p_rain = query_posterior(model, ["Rain"], {"WetGrass": 1})[1]

    # Query 2: P(Sprinkler=1 | WetGrass=1)
    p_sprinkler = query_posterior(model, ["Sprinkler"], {"WetGrass": 1})[1]

    # Query 3: P(Cloudy=1 | WetGrass=1)
    p_cloudy = query_posterior(model, ["Cloudy"], {"WetGrass": 1})[1]

    # Query 4: P(Rain=1 | WetGrass=1, Sprinkler=0)
    p_rain_sprinkler0 = query_posterior(model, ["Rain"], {"WetGrass": 1, "Sprinkler": 0})[1]

    # Verification via Enumeration Oracle
    p_rain_enum = compute_posterior_by_enumeration(model, "Rain", 1, {"WetGrass": 1})
    diff = abs(p_rain - p_rain_enum)

    return {
        "rain_given_wetgrass": float(p_rain),
        "sprinkler_given_wetgrass": float(p_sprinkler),
        "cloudy_given_wetgrass": float(p_cloudy),
        "rain_given_wetgrass_sprinkler0": float(p_rain_sprinkler0),
        "enumeration_rain_given_wetgrass": float(p_rain_enum),
        "enumeration_diff": float(diff),
    }
