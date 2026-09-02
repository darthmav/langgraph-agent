"""Closed-form tests for the spectral_graph package.

These assert against exact analytic spectra rather than against whatever the
code happens to produce:

- Path graph P_n:     lambda_k = 2 - 2*cos(pi*k/n),  k = 0..n-1
- Cycle graph C_n:    lambda_k = 2 - 2*cos(2*pi*k/n)
- Complete graph K_n: spectrum {0} + {n} with multiplicity n-1
- Fiedler value of P_n: lambda_2 = 2 - 2*cos(pi/n)

plus a cross-check of every unnormalized spectrum against NetworkX's
`laplacian_spectrum`, which is independent of this package's own code path.
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pytest

# The package is not installed; import it from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectral_graph import (  # noqa: E402
    adjacency_matrix,
    algebraic_connectivity,
    cheeger_bounds,
    compute_eigenpairs,
    compute_spectrum,
    conductance,
    degree_matrix,
    fiedler_vector,
    laplacian_matrix,
    normalized_laplacian_matrix,
    random_walk_laplacian_matrix,
    spectral_clustering,
    spectral_embedding,
    sweep_cut,
)

TOL = 1e-9


def path_spectrum(n: int) -> np.ndarray:
    """Analytic Laplacian spectrum of the path graph P_n."""
    return np.array([2 - 2 * np.cos(np.pi * k / n) for k in range(n)])


def cycle_spectrum(n: int) -> np.ndarray:
    """Analytic Laplacian spectrum of the cycle graph C_n."""
    return np.sort([2 - 2 * np.cos(2 * np.pi * k / n) for k in range(n)])


# ---------------------------------------------------------------------------
# Laplacian construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 9, 20])
def test_laplacian_equals_degree_minus_adjacency(n: int) -> None:
    G = nx.path_graph(n)
    L = laplacian_matrix(G).toarray()
    expected = degree_matrix(G).toarray() - adjacency_matrix(G).toarray()
    assert np.allclose(L, expected, atol=TOL)


@pytest.mark.parametrize("n", [3, 7, 15])
def test_laplacian_annihilates_constant_vector(n: int) -> None:
    """L @ 1 = 0 exactly: the constant vector spans the null space of a
    connected graph's Laplacian."""
    G = nx.path_graph(n)
    L = laplacian_matrix(G)
    assert np.allclose(L @ np.ones(n), 0.0, atol=TOL)


def test_normalized_laplacian_eigenvalues_lie_in_zero_two() -> None:
    for G in (nx.path_graph(12), nx.cycle_graph(9), nx.complete_graph(6),
              nx.karate_club_graph()):
        eigenvalues = compute_spectrum(G, normalized=True)
        assert eigenvalues.min() > -TOL
        assert eigenvalues.max() < 2.0 + 1e-9


def test_weighted_graph_uses_weighted_degrees() -> None:
    """A weighted graph must keep L = D - A consistent; otherwise the
    normalized Laplacian stops being positive semi-definite."""
    G = nx.karate_club_graph()  # carries edge weights
    assert np.allclose(
        laplacian_matrix(G).toarray(), nx.laplacian_matrix(G).toarray(), atol=TOL
    )
    L_norm = normalized_laplacian_matrix(G).toarray()
    assert np.allclose(
        L_norm, nx.normalized_laplacian_matrix(G).toarray(), atol=TOL
    )
    assert np.linalg.eigvalsh(L_norm).min() > -TOL


def test_random_walk_laplacian_shares_normalized_spectrum() -> None:
    """L_rw = I - D^-1 A is similar to L_sym, so the spectra coincide."""
    G = nx.cycle_graph(10)
    rw = np.sort(np.linalg.eigvals(random_walk_laplacian_matrix(G).toarray()).real)
    sym = np.sort(np.linalg.eigvalsh(normalized_laplacian_matrix(G).toarray()))
    assert np.allclose(rw, sym, atol=1e-8)


# ---------------------------------------------------------------------------
# Closed-form spectra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13, 30])
def test_path_graph_closed_form_spectrum(n: int) -> None:
    """P_n has lambda_k = 2 - 2*cos(pi*k/n) for k = 0..n-1."""
    G = nx.path_graph(n)
    eigenvalues = compute_spectrum(G)
    assert eigenvalues.shape == (n,)
    assert np.allclose(eigenvalues, path_spectrum(n), atol=1e-10)


@pytest.mark.parametrize("n", [3, 6, 11, 20])
def test_cycle_graph_closed_form_spectrum(n: int) -> None:
    """C_n has lambda_k = 2 - 2*cos(2*pi*k/n)."""
    G = nx.cycle_graph(n)
    eigenvalues = compute_spectrum(G)
    assert np.allclose(eigenvalues, cycle_spectrum(n), atol=1e-10)


@pytest.mark.parametrize("n", [2, 4, 7, 12])
def test_complete_graph_closed_form_spectrum(n: int) -> None:
    """K_n has spectrum {0} together with n repeated n-1 times."""
    G = nx.complete_graph(n)
    eigenvalues = compute_spectrum(G)

    assert eigenvalues.shape == (n,)
    assert abs(eigenvalues[0]) < 1e-10
    assert np.allclose(eigenvalues[1:], float(n), atol=1e-10)
    # Multiplicity of n is exactly n-1.
    assert int(np.sum(np.abs(eigenvalues - n) < 1e-10)) == n - 1


def test_disconnected_graph_zero_multiplicity_counts_components() -> None:
    """dim ker(L) = number of connected components."""
    G = nx.disjoint_union_all([nx.path_graph(4), nx.cycle_graph(5), nx.complete_graph(3)])
    eigenvalues = compute_spectrum(G)
    zeros = int(np.sum(np.abs(eigenvalues) < 1e-8))
    assert zeros == nx.number_connected_components(G) == 3


@pytest.mark.parametrize(
    "G",
    [
        nx.path_graph(9),
        nx.cycle_graph(9),
        nx.complete_graph(9),
        nx.star_graph(8),
        nx.barbell_graph(5, 2),
        nx.karate_club_graph(),
    ],
    ids=["path", "cycle", "complete", "star", "barbell", "karate"],
)
def test_spectrum_matches_networkx_laplacian_spectrum(G: nx.Graph) -> None:
    """Independent cross-check against NetworkX's own solver."""
    ours = compute_spectrum(G)
    theirs = np.sort(nx.laplacian_spectrum(G))
    assert np.allclose(ours, theirs, atol=1e-8)


def test_star_graph_closed_form_spectrum() -> None:
    """K_{1,n-1} has spectrum {0, 1 (multiplicity n-2), n}."""
    n = 8
    G = nx.star_graph(n - 1)
    eigenvalues = compute_spectrum(G)
    assert abs(eigenvalues[0]) < 1e-10
    assert np.allclose(eigenvalues[1:-1], 1.0, atol=1e-10)
    assert abs(eigenvalues[-1] - n) < 1e-10


# ---------------------------------------------------------------------------
# Algebraic connectivity and the Fiedler vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 10, 25])
def test_path_graph_fiedler_value_closed_form(n: int) -> None:
    """The Fiedler value of P_n is 2 - 2*cos(pi/n)."""
    G = nx.path_graph(n)
    expected = 2 - 2 * np.cos(np.pi / n)
    assert abs(algebraic_connectivity(G) - expected) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("n", [3, 6, 10])
def test_complete_graph_fiedler_value_is_n(n: int) -> None:
    G = nx.complete_graph(n)
    assert algebraic_connectivity(G) == pytest.approx(float(n), abs=1e-10)


def test_algebraic_connectivity_matches_networkx() -> None:
    for G in (nx.path_graph(11), nx.cycle_graph(11), nx.barbell_graph(5, 1)):
        assert algebraic_connectivity(G) == pytest.approx(
            nx.algebraic_connectivity(G), abs=1e-7
        )


def test_algebraic_connectivity_rejects_disconnected_graph() -> None:
    G = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
    with pytest.raises(ValueError, match="disconnected"):
        algebraic_connectivity(G)


def test_fiedler_vector_is_an_eigenvector_of_lambda_two() -> None:
    """L v = lambda_2 v, and v is orthogonal to the constant vector."""
    G = nx.path_graph(9)
    v = fiedler_vector(G)
    L = laplacian_matrix(G)
    lambda2 = algebraic_connectivity(G)

    assert np.allclose(L @ v, lambda2 * v, atol=1e-9)
    assert abs(float(np.sum(v))) < 1e-9
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-9)


def test_fiedler_vector_of_path_is_monotone() -> None:
    """On P_n the Fiedler vector is v_i = cos(pi*(i + 1/2)/n), monotone in i,
    so the sign cut splits the path in half."""
    n = 10
    G = nx.path_graph(n)
    v = fiedler_vector(G)
    if v[0] < 0:
        v = -v
    assert np.all(np.diff(v) < 0), f"not monotone: {v}"

    expected = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    expected /= np.linalg.norm(expected)
    assert np.allclose(v, expected, atol=1e-9)


def test_compute_eigenpairs_shapes_and_residuals() -> None:
    G = nx.cycle_graph(12)
    k = 4
    values, vectors = compute_eigenpairs(G, k=k)
    L = laplacian_matrix(G)

    assert values.shape == (k,)
    assert vectors.shape == (12, k)
    assert np.all(np.diff(values) >= -1e-12)  # ascending
    for i in range(k):
        assert np.allclose(L @ vectors[:, i], values[i] * vectors[:, i], atol=1e-9)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_spectral_embedding_shape_and_orthogonality() -> None:
    G = nx.path_graph(12)
    X = spectral_embedding(G, dim=3)
    assert X.shape == (12, 3)
    # Eigenvectors of a symmetric matrix are mutually orthonormal.
    assert np.allclose(X.T @ X, np.eye(3), atol=1e-9)
    # Each embedding column is orthogonal to the constant vector.
    assert np.allclose(X.sum(axis=0), 0.0, atol=1e-9)


def test_spectral_embedding_rejects_oversized_dimension() -> None:
    G = nx.path_graph(3)
    with pytest.raises(ValueError):
        spectral_embedding(G, dim=5)


# ---------------------------------------------------------------------------
# Clustering, conductance, Cheeger
# ---------------------------------------------------------------------------


def test_spectral_clustering_separates_two_cliques() -> None:
    G = nx.disjoint_union(nx.complete_graph(6), nx.complete_graph(6))
    G.add_edge(0, 6)
    labels = spectral_clustering(G, k=2)

    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_spectral_clustering_is_reproducible() -> None:
    G = nx.barbell_graph(6, 0)
    a = spectral_clustering(G, k=2, random_state=3)
    b = spectral_clustering(G, k=2, random_state=3)
    assert np.array_equal(a, b)


def test_conductance_closed_form_on_path() -> None:
    """Cutting P_4 in the middle: one crossing edge, volume 3 on each side."""
    G = nx.path_graph(4)
    assert conductance(G, {0, 1}) == pytest.approx(1 / 3, abs=1e-12)
    assert conductance(G, {2, 3}) == pytest.approx(1 / 3, abs=1e-12)
    assert conductance(G, set()) == float("inf")


def test_sweep_cut_finds_the_bridge_of_a_barbell() -> None:
    """Barbell(5,0): the sweep cut is a whole bell, conductance 1/21."""
    G = nx.barbell_graph(5, 0)
    S, phi = sweep_cut(G)
    assert sorted(S) in ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    assert phi == pytest.approx(1 / 21, abs=1e-12)


@pytest.mark.parametrize(
    "G",
    [
        nx.path_graph(12),
        nx.cycle_graph(12),
        nx.barbell_graph(6, 0),
        nx.karate_club_graph(),
    ],
    ids=["path", "cycle", "barbell", "karate"],
)
def test_cheeger_inequality_brackets_the_sweep_cut(G: nx.Graph) -> None:
    """lambda_2/2 <= phi(sweep cut) <= sqrt(2*lambda_2)."""
    lower, upper = cheeger_bounds(G)
    _, phi = sweep_cut(G)
    assert lower - 1e-9 <= phi <= upper + 1e-9


def test_cheeger_bounds_closed_form_on_complete_graph() -> None:
    """K_n has normalized lambda_2 = n/(n-1)."""
    n = 8
    lower, upper = cheeger_bounds(nx.complete_graph(n))
    lambda2 = n / (n - 1)
    assert lower == pytest.approx(lambda2 / 2, abs=1e-9)
    assert upper == pytest.approx(np.sqrt(2 * lambda2), abs=1e-9)
