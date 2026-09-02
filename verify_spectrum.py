#!/usr/bin/env python
"""Root-level verification script for spectral_graph.spectrum module."""

import sys
import os

# Insert project root into sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
from spectral_graph.laplacian import laplacian_matrix, normalized_laplacian_matrix
from spectral_graph.spectrum import compute_spectrum, compute_eigenpairs, algebraic_connectivity

print("Testing spectrum.py...")

# Test on path graph
G = nx.path_graph(5)

# Test compute_spectrum
eigenvalues = compute_spectrum(G)
print(f"All eigenvalues: {eigenvalues}")
assert len(eigenvalues) == 5
assert abs(eigenvalues[0]) < 1e-10  # First eigenvalue is 0
print("✓ compute_spectrum works correctly")

# Test compute_eigenpairs
evals, evecs = compute_eigenpairs(G, k=3)
print(f"First 3 eigenvalues: {evals}")
print(f"Eigenvector matrix shape: {evecs.shape}")
assert evals.shape == (3,)
assert evecs.shape == (5, 3)
print("✓ compute_eigenpairs works correctly")

# Test algebraic_connectivity
ac = algebraic_connectivity(G)
print(f"Algebraic connectivity: {ac:.4f}")
# For path graph P_n, λ₂ = 2(1 - cos(π/n))
expected = 2 * (1 - np.cos(np.pi / 5))
assert abs(ac - expected) < 1e-6, f"Expected {expected}, got {ac}"
print("✓ algebraic_connectivity matches theoretical value for path graph")

# Cross-check with NetworkX
ac_nx = nx.algebraic_connectivity(G)
assert abs(ac - ac_nx) < 1e-6, f"Mismatch with NetworkX: {ac} vs {ac_nx}"
print("✓ algebraic_connectivity matches NetworkX")

# Test on larger graph with sparse solver
G_large = nx.barabasi_albert_graph(100, 3, seed=42)
eigenvalues_sparse = compute_spectrum(G_large, k=5)
print(f"5 smallest eigenvalues of BA graph: {eigenvalues_sparse}")
assert len(eigenvalues_sparse) == 5
print("✓ Sparse solver works for larger graphs")

print("\nAll spectrum.py tests passed!")
