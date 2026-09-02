"""
Laplacian matrix constructions for spectral graph theory.

Provides functions to compute:
- Unnormalized Laplacian: L = D - A
- Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
- Random walk Laplacian: L_rw = I - D^(-1) A

All functions accept NetworkX graphs and return scipy sparse matrices.
"""

import networkx as nx
import numpy as np
from scipy import sparse


def adjacency_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """
    Compute the adjacency matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (can be weighted or unweighted)

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix as a sparse CSR matrix

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import adjacency_matrix
    >>> G = nx.path_graph(5)
    >>> A = adjacency_matrix(G)
    >>> A.shape
    (5, 5)
    """
    return nx.adjacency_matrix(G).astype(np.float64)


def degree_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """
    Compute the degree matrix of a graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    Returns
    -------
    scipy.sparse.csr_matrix
        Diagonal degree matrix as a sparse CSR matrix

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import degree_matrix
    >>> G = nx.path_graph(5)
    >>> D = degree_matrix(G)
    >>> D.diagonal()
    array([1., 2., 2., 2., 1.])
    """
    # Weight-aware to match adjacency_matrix, which honours edge weights by
    # default. Mixing a weighted A with unweighted degrees makes L = D - A
    # non-PSD on any weighted graph (nx.karate_club_graph() is one).
    degrees = np.array([d for _, d in G.degree(weight="weight")], dtype=np.float64)
    return sparse.diags(degrees, format="csr")


def laplacian_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """
    Compute the unnormalized Laplacian matrix L = D - A.

    The Laplacian is positive semi-definite with eigenvalues
    0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ. The multiplicity of eigenvalue 0
    equals the number of connected components.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    Returns
    -------
    scipy.sparse.csr_matrix
        Unnormalized Laplacian matrix

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import laplacian_matrix
    >>> G = nx.path_graph(5)
    >>> L = laplacian_matrix(G)
    >>> L.shape
    (5, 5)
    >>> # Verify L @ [1,1,1,1,1] = 0 (constant vector is in null space)
    >>> import numpy as np
    >>> ones = np.ones(5)
    >>> np.allclose(L @ ones, 0)
    True
    """
    A = adjacency_matrix(G)
    D = degree_matrix(G)
    return D - A


def normalized_laplacian_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """
    Compute the normalized Laplacian L_norm = I - D^(-1/2) A D^(-1/2).

    Also known as the symmetric normalized Laplacian. Eigenvalues lie
    in the interval [0, 2]. Better suited for graphs with heterogeneous
    degree distributions.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    Returns
    -------
    scipy.sparse.csr_matrix
        Normalized Laplacian matrix

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import normalized_laplacian_matrix
    >>> G = nx.path_graph(5)
    >>> L_norm = normalized_laplacian_matrix(G)
    >>> L_norm.shape
    (5, 5)
    """
    n = G.number_of_nodes()
    A = adjacency_matrix(G)
    degrees = np.array([d for _, d in G.degree(weight="weight")], dtype=np.float64)

    # Handle isolated nodes (degree 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    D_inv_sqrt = sparse.diags(d_inv_sqrt, format="csr")
    I = sparse.eye(n, format="csr")

    # L_norm = I - D^(-1/2) A D^(-1/2)
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def random_walk_laplacian_matrix(G: nx.Graph) -> sparse.csr_matrix:
    """
    Compute the random walk Laplacian L_rw = I - D^(-1) A.

    Also known as the asymmetric normalized Laplacian. Shares eigenvalues
    with the symmetric normalized Laplacian but is not symmetric.
    Used in random walk analysis and PageRank.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    Returns
    -------
    scipy.sparse.csr_matrix
        Random walk Laplacian matrix

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import random_walk_laplacian_matrix
    >>> G = nx.path_graph(5)
    >>> L_rw = random_walk_laplacian_matrix(G)
    >>> L_rw.shape
    (5, 5)
    """
    n = G.number_of_nodes()
    A = adjacency_matrix(G)
    degrees = np.array([d for _, d in G.degree(weight="weight")], dtype=np.float64)

    # Handle isolated nodes (degree 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv = 1.0 / degrees
    d_inv[np.isinf(d_inv)] = 0.0

    D_inv = sparse.diags(d_inv, format="csr")
    I = sparse.eye(n, format="csr")

    # L_rw = I - D^(-1) A
    return I - D_inv @ A


if __name__ == "__main__":
    # Run simple validation
    import networkx as nx

    print("Testing laplacian.py...")

    # Test on path graph
    G = nx.path_graph(5)

    A = adjacency_matrix(G)
    print(f"Adjacency matrix shape: {A.shape}")

    D = degree_matrix(G)
    print(f"Degree matrix diagonal: {D.diagonal()}")

    L = laplacian_matrix(G)
    print(f"Laplacian matrix shape: {L.shape}")

    # Verify L @ 1 = 0
    ones = np.ones(G.number_of_nodes())
    assert np.allclose(L @ ones, 0), "Laplacian should annihilate constant vector"
    print("✓ L @ 1 = 0 verified")

    L_norm = normalized_laplacian_matrix(G)
    print(f"Normalized Laplacian shape: {L_norm.shape}")

    L_rw = random_walk_laplacian_matrix(G)
    print(f"Random walk Laplacian shape: {L_rw.shape}")

    # Cross-check with NetworkX
    L_nx = nx.laplacian_matrix(G).toarray()
    assert np.allclose(L.toarray(), L_nx), "Laplacian mismatch with NetworkX"
    print("✓ Laplacian matches NetworkX")

    L_norm_nx = nx.normalized_laplacian_matrix(G).toarray()
    assert np.allclose(L_norm.toarray(), L_norm_nx), "Normalized Laplacian mismatch"
    print("✓ Normalized Laplacian matches NetworkX")

    # Weighted graph: A is weight-aware, so the degrees must be too, or the
    # normalized Laplacian stops being positive semi-definite.
    G_w = nx.karate_club_graph()  # carries edge weights
    L_w = laplacian_matrix(G_w)
    assert np.allclose(L_w.toarray(), nx.laplacian_matrix(G_w).toarray()), (
        "Weighted Laplacian mismatch with NetworkX"
    )
    L_w_norm = normalized_laplacian_matrix(G_w).toarray()
    assert np.allclose(L_w_norm, nx.normalized_laplacian_matrix(G_w).toarray()), (
        "Weighted normalized Laplacian mismatch with NetworkX"
    )
    assert np.linalg.eigvalsh(L_w_norm)[0] > -1e-9, (
        "Normalized Laplacian is not positive semi-definite"
    )
    print("✓ Weighted graph matches NetworkX and stays PSD")

    print("\nAll laplacian.py tests passed!")
