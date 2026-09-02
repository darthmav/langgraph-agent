#!/usr/bin/env python3
"""
Root-level test script for spectral_graph package.

This script verifies that all spectral_graph modules can be imported
and performs basic instantiation tests to prove the package structure is viable.

Scope: Import verification and basic instantiation tests only.
Full functionality testing is out of scope for "idealizing" deliverable.
"""

import sys


def test_imports():
    """Test that all spectral_graph modules can be imported."""
    print("Testing imports...")
    
    # Test package-level imports
    from spectral_graph import (
        # Laplacian construction
        laplacian_matrix,
        normalized_laplacian_matrix,
        random_walk_laplacian_matrix,
        degree_matrix,
        adjacency_matrix,
        # Spectrum computation
        compute_spectrum,
        compute_eigenpairs,
        algebraic_connectivity,
        # Fiedler vector
        fiedler_vector,
        fiedler_partition,
        spectral_bipartition,
        # Embedding
        spectral_embedding,
        laplacian_eigenmap,
        embed_and_normalize,
        # Clustering and conductance
        spectral_clustering,
        conductance,
        sweep_cut,
        cheeger_bounds,
    )
    print("  ✓ Package-level imports successful")
    
    # Test module-level imports
    from spectral_graph import laplacian
    from spectral_graph import spectrum
    from spectral_graph import fiedler
    from spectral_graph import embedding
    from spectral_graph import clustering
    from spectral_graph import stability
    from spectral_graph import operations
    print("  ✓ Module-level imports successful")
    
    return True


def test_basic_instantiation():
    """Test basic instantiation of key classes/functions."""
    print("\nTesting basic instantiation...")
    
    import networkx as nx
    import numpy as np
    
    # Create a simple test graph
    G = nx.path_graph(5)
    print(f"  Created test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test laplacian module
    from spectral_graph.laplacian import (
        adjacency_matrix,
        degree_matrix,
        laplacian_matrix,
        normalized_laplacian_matrix,
        random_walk_laplacian_matrix,
    )
    
    A = adjacency_matrix(G)
    assert A.shape == (5, 5), f"Expected (5, 5), got {A.shape}"
    print("  ✓ adjacency_matrix works")
    
    D = degree_matrix(G)
    assert D.shape == (5, 5), f"Expected (5, 5), got {D.shape}"
    print("  ✓ degree_matrix works")
    
    L = laplacian_matrix(G)
    assert L.shape == (5, 5), f"Expected (5, 5), got {L.shape}"
    print("  ✓ laplacian_matrix works")
    
    L_norm = normalized_laplacian_matrix(G)
    assert L_norm.shape == (5, 5), f"Expected (5, 5), got {L_norm.shape}"
    print("  ✓ normalized_laplacian_matrix works")
    
    L_rw = random_walk_laplacian_matrix(G)
    assert L_rw.shape == (5, 5), f"Expected (5, 5), got {L_rw.shape}"
    print("  ✓ random_walk_laplacian_matrix works")
    
    # Test spectrum module
    from spectral_graph.spectrum import (
        compute_spectrum,
        compute_eigenpairs,
        algebraic_connectivity,
    )
    
    eigenvalues = compute_spectrum(G)
    assert len(eigenvalues) == 5, f"Expected 5 eigenvalues, got {len(eigenvalues)}"
    print("  ✓ compute_spectrum works")
    
    evals, evecs = compute_eigenpairs(G, k=2)
    assert evals.shape == (2,), f"Expected (2,), got {evals.shape}"
    assert evecs.shape == (5, 2), f"Expected (5, 2), got {evecs.shape}"
    print("  ✓ compute_eigenpairs works")
    
    ac = algebraic_connectivity(G)
    assert isinstance(ac, float), f"Expected float, got {type(ac)}"
    print("  ✓ algebraic_connectivity works")
    
    # Test fiedler module
    from spectral_graph.fiedler import (
        fiedler_vector,
        fiedler_partition,
        spectral_bipartition,
    )
    
    fv = fiedler_vector(G)
    assert fv.shape == (5,), f"Expected (5,), got {fv.shape}"
    print("  ✓ fiedler_vector works")
    
    set1, set2 = fiedler_partition(G)
    assert len(set1) + len(set2) == 5, "Partition should cover all nodes"
    print("  ✓ fiedler_partition works")
    
    result = spectral_bipartition(G)
    assert 'set1' in result and 'set2' in result, "Missing keys in result"
    print("  ✓ spectral_bipartition works")
    
    # Test embedding module
    from spectral_graph.embedding import (
        spectral_embedding,
        laplacian_eigenmap,
        embed_and_normalize,
    )
    
    emb = spectral_embedding(G, dim=2)
    assert emb.shape == (5, 2), f"Expected (5, 2), got {emb.shape}"
    print("  ✓ spectral_embedding works")
    
    le = laplacian_eigenmap(G, dim=2)
    assert le.shape == (5, 2), f"Expected (5, 2), got {le.shape}"
    print("  ✓ laplacian_eigenmap works")
    
    en = embed_and_normalize(G, dim=2)
    assert en.shape == (5, 2), f"Expected (5, 2), got {en.shape}"
    print("  ✓ embed_and_normalize works")
    
    # Test clustering module
    from spectral_graph.clustering import (
        spectral_clustering,
        conductance,
        sweep_cut,
        cheeger_bounds,
    )
    
    labels = spectral_clustering(G, k=2)
    assert labels.shape == (5,), f"Expected (5,), got {labels.shape}"
    print("  ✓ spectral_clustering works")
    
    phi = conductance(G, {0, 1})
    assert isinstance(phi, float), f"Expected float, got {type(phi)}"
    print("  ✓ conductance works")
    
    S, phi_sweep = sweep_cut(G)
    assert isinstance(S, set), f"Expected set, got {type(S)}"
    print("  ✓ sweep_cut works")
    
    lower, upper = cheeger_bounds(G)
    assert lower <= upper, "Lower bound should be <= upper bound"
    print("  ✓ cheeger_bounds works")
    
    # Test stability module
    from spectral_graph.stability import (
        check_condition_number,
        check_eigenvalue_stability,
        choose_eigen_solver,
        verify_psd,
        safe_divide,
        safe_sqrt_inverse,
    )
    
    is_stable, cond = check_condition_number(np.eye(3))
    assert is_stable, "Identity matrix should be stable"
    print("  ✓ check_condition_number works")
    
    is_stable, diag = check_eigenvalue_stability(np.array([0.0, 0.5, 1.0]))
    assert is_stable, "Non-negative eigenvalues should be stable"
    print("  ✓ check_eigenvalue_stability works")
    
    solver = choose_eigen_solver(30, 5)
    assert solver in ('dense', 'sparse'), f"Expected 'dense' or 'sparse', got {solver}"
    print("  ✓ choose_eigen_solver works")
    
    is_psd, min_eval = verify_psd(np.eye(3))
    assert is_psd, "Identity matrix should be PSD"
    print("  ✓ verify_psd works")
    
    result = safe_divide(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert result[0] == 1.0 and result[1] == 0.0, "safe_divide failed"
    print("  ✓ safe_divide works")
    
    result = safe_sqrt_inverse(np.array([1.0, 4.0, 0.0]))
    assert np.allclose(result, [1.0, 0.5, 0.0]), "safe_sqrt_inverse failed"
    print("  ✓ safe_sqrt_inverse works")
    
    # Test operations module
    from spectral_graph.operations import (
        compute_spectrum_stable,
        spectral_filter,
        laplacian_add,
        laplacian_scale,
        laplacian_convex_combination,
        spectral_distance,
        cheeger_constant_estimate,
    )
    
    eigenvalues, diag = compute_spectrum_stable(G, check_stability=True)
    assert len(eigenvalues) == 5, f"Expected 5 eigenvalues, got {len(eigenvalues)}"
    print("  ✓ compute_spectrum_stable works")
    
    signal = np.random.randn(5)
    lowpass = lambda x: np.exp(-x / 2.0)
    filtered = spectral_filter(G, signal, lowpass)
    assert filtered.shape == (5,), f"Expected (5,), got {filtered.shape}"
    print("  ✓ spectral_filter works")
    
    G2 = nx.cycle_graph(5)
    L1 = laplacian_matrix(G)
    L2 = laplacian_matrix(G2)
    L_sum = laplacian_add(L1, L2)
    assert L_sum.shape == (5, 5), f"Expected (5, 5), got {L_sum.shape}"
    print("  ✓ laplacian_add works")
    
    L_scaled = laplacian_scale(L1, 2.0)
    assert L_scaled.shape == (5, 5), f"Expected (5, 5), got {L_scaled.shape}"
    print("  ✓ laplacian_scale works")
    
    L_combined = laplacian_convex_combination(L1, L2, 0.5)
    assert L_combined.shape == (5, 5), f"Expected (5, 5), got {L_combined.shape}"
    print("  ✓ laplacian_convex_combination works")
    
    dist = spectral_distance(G, G)
    assert isinstance(dist, float), f"Expected float, got {type(dist)}"
    print("  ✓ spectral_distance works")
    
    lower, upper = cheeger_constant_estimate(G)
    assert lower <= upper, "Lower bound should be <= upper bound"
    print("  ✓ cheeger_constant_estimate works")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("spectral_graph Package Verification")
    print("=" * 60)
    
    try:
        # Test 1: Import verification
        if not test_imports():
            print("\n❌ Import tests failed")
            sys.exit(1)
        
        # Test 2: Basic instantiation
        if not test_basic_instantiation():
            print("\n❌ Instantiation tests failed")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("All tests passed! Package structure is viable.")
        print("=" * 60)
        return 0
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
