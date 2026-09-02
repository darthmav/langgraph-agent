#!/usr/bin/env python3
"""
Verification script for spectral_graph.clustering module.

This script must be run from the project root. It inserts the project root
into sys.path and imports spectral_graph absolutely to verify the clustering
module works correctly.
"""

import os
import sys

# Insert project root into sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import networkx as nx
import numpy as np

from spectral_graph import (
    cheeger_bounds,
    conductance,
    spectral_clustering,
    sweep_cut,
)


def _two_cliques(size=5):
    """Two cliques joined by a single bridge edge."""
    G = nx.disjoint_union(nx.complete_graph(size), nx.complete_graph(size))
    G.add_edge(0, size)
    return G


def test_spectral_clustering_two_cliques():
    """Two cliques joined by a bridge must split along the bridge."""
    G = _two_cliques(5)
    labels = spectral_clustering(G, k=2)

    assert labels.shape == (10,), f"Expected shape (10,), got {labels.shape}"
    assert len(set(labels[:5])) == 1, f"First clique split: {labels[:5]}"
    assert len(set(labels[5:])) == 1, f"Second clique split: {labels[5:]}"
    assert labels[0] != labels[5], "Both cliques landed in the same cluster"
    print("✓ test_spectral_clustering_two_cliques passed")


def test_spectral_clustering_three_blocks():
    """Three weakly coupled cliques recover the planted blocks."""
    G = nx.disjoint_union_all([nx.complete_graph(6) for _ in range(3)])
    G.add_edge(0, 6)
    G.add_edge(6, 12)
    labels = spectral_clustering(G, k=3)

    assert labels.shape == (18,)
    for block in range(3):
        segment = labels[block * 6:(block + 1) * 6]
        assert len(set(segment)) == 1, f"Block {block} split: {segment}"
    assert len(set(labels)) == 3, f"Expected 3 clusters, got {set(labels)}"
    print("✓ test_spectral_clustering_three_blocks passed")


def test_spectral_clustering_deterministic():
    """The same random_state gives the same labels."""
    G = _two_cliques(5)
    first = spectral_clustering(G, k=2, random_state=7)
    second = spectral_clustering(G, k=2, random_state=7)

    assert np.array_equal(first, second), "Clustering is not reproducible"
    print("✓ test_spectral_clustering_deterministic passed")


def test_spectral_clustering_k_one_and_errors():
    """k=1 is trivial; k<1 and k>n are rejected."""
    G = nx.path_graph(5)

    labels = spectral_clustering(G, k=1)
    assert np.array_equal(labels, np.zeros(5, dtype=int)), "k=1 must be one cluster"

    for bad_k in (0, 6):
        try:
            spectral_clustering(G, k=bad_k)
            assert False, f"k={bad_k} should have raised ValueError"
        except ValueError:
            pass
    print("✓ test_spectral_clustering_k_one_and_errors passed")


def test_conductance_closed_form():
    """Conductance of the middle cut of P_4 is exactly 1/3."""
    G = nx.path_graph(4)
    # vol({0,1}) = 1 + 2 = 3, vol({2,3}) = 3, one crossing edge.
    phi = conductance(G, {0, 1})
    assert abs(phi - 1.0 / 3.0) < 1e-12, f"Expected 1/3, got {phi}"

    # Symmetric in the complement.
    assert abs(conductance(G, {2, 3}) - phi) < 1e-12, "Conductance not symmetric"

    # Degenerate sets have no finite conductance.
    assert conductance(G, set()) == float("inf")
    assert conductance(G, {0, 1, 2, 3}) == float("inf")
    print("✓ test_conductance_closed_form passed")


def test_conductance_weighted():
    """Edge weights are honoured."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=0.5)
    G.add_edge(2, 3, weight=2.0)
    # vol({0,1}) = 2 + 2.5 = 4.5, vol({2,3}) = 4.5, crossing weight 0.5
    phi = conductance(G, {0, 1})
    assert abs(phi - 0.5 / 4.5) < 1e-12, f"Expected {0.5 / 4.5}, got {phi}"
    print("✓ test_conductance_weighted passed")


def test_sweep_cut_barbell():
    """The sweep cut of a barbell recovers one of the two bells."""
    G = nx.barbell_graph(5, 0)
    S, phi = sweep_cut(G)

    assert sorted(S) in ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), f"Unexpected cut: {sorted(S)}"
    # One crossing edge, vol = 5*4 + 1 = 21 on each side.
    assert abs(phi - 1.0 / 21.0) < 1e-12, f"Expected 1/21, got {phi}"
    # The reported conductance agrees with the standalone function.
    assert abs(conductance(G, S) - phi) < 1e-12, "sweep_cut and conductance disagree"
    print("✓ test_sweep_cut_barbell passed")


def test_sweep_cut_small_graph_raises():
    """A single-node graph has no cut."""
    G = nx.Graph()
    G.add_node(0)
    try:
        sweep_cut(G)
        assert False, "Should have raised ValueError for a 1-node graph"
    except ValueError as e:
        assert "2 nodes" in str(e)
    print("✓ test_sweep_cut_small_graph_raises passed")


def test_cheeger_inequality_holds():
    """lambda_2/2 <= phi(sweep cut) <= sqrt(2*lambda_2) on several graphs."""
    graphs = {
        "barbell(6,0)": nx.barbell_graph(6, 0),
        "path(12)": nx.path_graph(12),
        "cycle(12)": nx.cycle_graph(12),
        "karate": nx.karate_club_graph(),
        "two cliques": _two_cliques(7),
    }
    for name, G in graphs.items():
        lower, upper = cheeger_bounds(G)
        _, phi = sweep_cut(G)
        assert lower - 1e-9 <= phi <= upper + 1e-9, (
            f"{name}: Cheeger violated, {lower} <= {phi} <= {upper} is false"
        )
        print(f"    {name}: {lower:.4f} <= phi={phi:.4f} <= {upper:.4f}")
    print("✓ test_cheeger_inequality_holds passed")


def test_cheeger_bounds_complete_graph():
    """K_n has normalized lambda_2 = n/(n-1), so the bounds are closed-form."""
    n = 8
    G = nx.complete_graph(n)
    lower, upper = cheeger_bounds(G)

    lambda2 = n / (n - 1)
    assert abs(lower - lambda2 / 2) < 1e-9, f"Expected {lambda2 / 2}, got {lower}"
    assert abs(upper - np.sqrt(2 * lambda2)) < 1e-9, f"Expected {np.sqrt(2 * lambda2)}, got {upper}"
    print("✓ test_cheeger_bounds_complete_graph passed")


if __name__ == "__main__":
    print("Running Clustering module verification tests...\n")

    test_spectral_clustering_two_cliques()
    test_spectral_clustering_three_blocks()
    test_spectral_clustering_deterministic()
    test_spectral_clustering_k_one_and_errors()
    test_conductance_closed_form()
    test_conductance_weighted()
    test_sweep_cut_barbell()
    test_sweep_cut_small_graph_raises()
    test_cheeger_inequality_holds()
    test_cheeger_bounds_complete_graph()

    print("\n✓ All Clustering verification tests passed!")
