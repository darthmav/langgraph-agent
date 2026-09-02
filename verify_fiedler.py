#!/usr/bin/env python3
"""
Verification script for spectral_graph.fiedler module.

This script must be run from the project root. It inserts the project root
into sys.path and imports spectral_graph absolutely to verify the fiedler
module works correctly.
"""

import sys
import os

# Insert project root into sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import networkx as nx
from spectral_graph import fiedler_vector, fiedler_partition, spectral_bipartition


def test_fiedler_vector_path_graph():
    """Test Fiedler vector on a path graph."""
    G = nx.path_graph(5)
    fiedler = fiedler_vector(G)
    
    assert fiedler.shape == (5,), f"Expected shape (5,), got {fiedler.shape}"
    # Fiedler vector should be orthogonal to constant vector (sum to ~0)
    assert np.abs(np.sum(fiedler)) < 1e-10, f"Fiedler vector should sum to 0, got {np.sum(fiedler)}"
    print("✓ test_fiedler_vector_path_graph passed")


def test_fiedler_vector_complete_graph():
    """Test Fiedler vector on a complete graph."""
    G = nx.complete_graph(4)
    fiedler = fiedler_vector(G)
    
    assert fiedler.shape == (4,), f"Expected shape (4,), got {fiedler.shape}"
    print("✓ test_fiedler_vector_complete_graph passed")


def test_fiedler_partition():
    """Test Fiedler partition on a path graph."""
    G = nx.path_graph(6)
    set1, set2 = fiedler_partition(G)
    
    assert len(set1) + len(set2) == 6, "Partition should include all nodes"
    assert len(set1) > 0 and len(set2) > 0, "Both partitions should be non-empty"
    assert set1.isdisjoint(set2), "Partitions should be disjoint"
    print("✓ test_fiedler_partition passed")


def test_spectral_bipartition():
    """Test spectral bipartition on a path graph."""
    G = nx.path_graph(5)
    result = spectral_bipartition(G)
    
    assert "set1" in result and "set2" in result, "Result should contain set1 and set2"
    assert "cut_size" in result, "Result should contain cut_size"
    assert "balance" in result, "Result should contain balance"
    assert "fiedler_value" in result, "Result should contain fiedler_value"
    
    # For path graph P_5, the natural cut is in the middle with cut_size=1
    assert result["cut_size"] >= 1, "Cut size should be at least 1"
    assert 0 < result["balance"] <= 1, "Balance should be between 0 and 1"
    print("✓ test_spectral_bipartition passed")


def test_normalized_laplacian():
    """Test Fiedler vector with normalized Laplacian."""
    G = nx.path_graph(5)
    fiedler_unnorm = fiedler_vector(G, normalized=False)
    fiedler_norm = fiedler_vector(G, normalized=True)
    
    assert fiedler_unnorm.shape == fiedler_norm.shape, "Shapes should match"
    print("✓ test_normalized_laplacian passed")


def test_disconnected_graph_raises():
    """Test that disconnected graph raises ValueError."""
    G = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
    
    try:
        fiedler_vector(G)
        assert False, "Should have raised ValueError for disconnected graph"
    except ValueError as e:
        assert "connected" in str(e).lower()
        print("✓ test_disconnected_graph_raises passed")


def test_small_graph_raises():
    """Test that graph with < 2 nodes raises ValueError."""
    G = nx.Graph()
    G.add_node(0)
    
    try:
        fiedler_vector(G)
        assert False, "Should have raised ValueError for graph with < 2 nodes"
    except ValueError as e:
        assert "2 nodes" in str(e)
        print("✓ test_small_graph_raises passed")


if __name__ == "__main__":
    print("Running Fiedler module verification tests...\n")
    
    test_fiedler_vector_path_graph()
    test_fiedler_vector_complete_graph()
    test_fiedler_partition()
    test_spectral_bipartition()
    test_normalized_laplacian()
    test_disconnected_graph_raises()
    test_small_graph_raises()
    
    print("\n✓ All Fiedler verification tests passed!")
