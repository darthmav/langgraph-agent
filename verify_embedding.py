#!/usr/bin/env python3
"""
Verification script for spectral_graph.embedding module.

This script must be run from the project root. It inserts the project root
into sys.path and imports spectral_graph absolutely to verify the embedding
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
from spectral_graph import spectral_embedding, laplacian_eigenmap, embed_and_normalize


def test_spectral_embedding_shape():
    """Test that spectral embedding returns correct shape."""
    G = nx.path_graph(10)
    embedding = spectral_embedding(G, dim=2)
    
    assert embedding.shape == (10, 2), f"Expected shape (10, 2), got {embedding.shape}"
    print("✓ test_spectral_embedding_shape passed")


def test_spectral_embedding_dim_3():
    """Test 3D spectral embedding."""
    G = nx.path_graph(10)
    embedding = spectral_embedding(G, dim=3)
    
    assert embedding.shape == (10, 3), f"Expected shape (10, 3), got {embedding.shape}"
    print("✓ test_spectral_embedding_dim_3 passed")


def test_spectral_embedding_normalized():
    """Test spectral embedding with normalized Laplacian."""
    G = nx.path_graph(10)
    embedding_unnorm = spectral_embedding(G, dim=2, normalized=False)
    embedding_norm = spectral_embedding(G, dim=2, normalized=True)
    
    assert embedding_unnorm.shape == embedding_norm.shape, "Shapes should match"
    # They should be different (different Laplacians)
    assert not np.allclose(embedding_unnorm, embedding_norm), "Normalized and unnormalized should differ"
    print("✓ test_spectral_embedding_normalized passed")


def test_laplacian_eigenmap():
    """Test Laplacian Eigenmap embedding."""
    G = nx.karate_club_graph()
    embedding = laplacian_eigenmap(G, dim=2)
    
    assert embedding.shape == (34, 2), f"Expected shape (34, 2), got {embedding.shape}"
    print("✓ test_laplacian_eigenmap passed")


def test_embed_and_normalize():
    """Test that embed_and_normalize returns unit-norm rows."""
    G = nx.path_graph(10)
    embedding = embed_and_normalize(G, dim=2)
    
    # Check each row has unit norm
    row_norms = np.linalg.norm(embedding, axis=1)
    assert np.allclose(row_norms, 1.0), f"Row norms should be 1.0, got {row_norms}"
    print("✓ test_embed_and_normalize passed")


def test_spectral_embedding_complete_graph():
    """Test spectral embedding on complete graph."""
    G = nx.complete_graph(6)
    embedding = spectral_embedding(G, dim=2)
    
    assert embedding.shape == (6, 2), f"Expected shape (6, 2), got {embedding.shape}"
    print("✓ test_spectral_embedding_complete_graph passed")


def test_spectral_embedding_small_graph():
    """Test spectral embedding on small graph."""
    G = nx.path_graph(4)
    embedding = spectral_embedding(G, dim=2)
    
    assert embedding.shape == (4, 2), f"Expected shape (4, 2), got {embedding.shape}"
    print("✓ test_spectral_embedding_small_graph passed")


def test_embedding_without_fiedler_skip():
    """Test embedding without skipping first eigenvector."""
    G = nx.path_graph(10)
    embedding = spectral_embedding(G, dim=2, use_fiedler=False)
    
    assert embedding.shape == (10, 2), f"Expected shape (10, 2), got {embedding.shape}"
    print("✓ test_embedding_without_fiedler_skip passed")


if __name__ == "__main__":
    print("Running Embedding module verification tests...\n")
    
    test_spectral_embedding_shape()
    test_spectral_embedding_dim_3()
    test_spectral_embedding_normalized()
    test_laplacian_eigenmap()
    test_embed_and_normalize()
    test_spectral_embedding_complete_graph()
    test_spectral_embedding_small_graph()
    test_embedding_without_fiedler_skip()
    
    print("\n✓ All Embedding verification tests passed!")
