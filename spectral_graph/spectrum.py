"""
Spectrum computation for spectral graph theory.

Provides functions to compute eigenvalues and eigenvectors of Laplacian
matrices using appropriate solvers (dense vs sparse) based on graph size.
"""

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from spectral_graph.laplacian import laplacian_matrix, normalized_laplacian_matrix


def compute_spectrum(
    G: nx.Graph,
    k: int = None,
    normalized: bool = False,
    which: str = "SM",
) -> np.ndarray:
    """
    Compute eigenvalues of the graph Laplacian.

    Automatically chooses dense or sparse solver based on graph size.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    k : int, optional
        Number of eigenvalues to compute. If None, computes all eigenvalues.
    normalized : bool, default False
        If True, use normalized Laplacian; otherwise use unnormalized.
    which : str, default 'SM'
        Which eigenvalues to compute (for sparse solver):
        - 'SM': smallest magnitude
        - 'LM': largest magnitude
        - 'SA': smallest algebraic
        - 'LA': largest algebraic

    Returns
    -------
    numpy.ndarray
        Array of eigenvalues in ascending order

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import compute_spectrum
    >>> G = nx.path_graph(5)
    >>> eigenvalues = compute_spectrum(G)
    >>> eigenvalues.shape
    (5,)
    >>> # First eigenvalue should be 0 (or very close)
    >>> abs(eigenvalues[0]) < 1e-10
    True
    """
    n = G.number_of_nodes()

    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)

    # Use dense solver for small graphs, sparse for large
    if k is None or k >= n - 1 or n < 50:
        # Dense solver - compute all eigenvalues
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues = np.linalg.eigvalsh(L_dense)
    else:
        # Sparse solver - compute k eigenvalues
        eigenvalues = eigsh(L, k=k, which=which, return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)

    return eigenvalues


def compute_eigenpairs(
    G: nx.Graph,
    k: int = 2,
    normalized: bool = False,
    which: str = "SM",
) -> tuple:
    """
    Compute k eigenvalue-eigenvector pairs of the graph Laplacian.

    Automatically chooses dense or sparse solver based on graph size.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    k : int, default 2
        Number of eigenpairs to compute
    normalized : bool, default False
        If True, use normalized Laplacian; otherwise use unnormalized.
    which : str, default 'SM'
        Which eigenvalues to compute (for sparse solver):
        - 'SM': smallest magnitude
        - 'LM': largest magnitude
        - 'SA': smallest algebraic
        - 'LA': largest algebraic

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors) where:
        - eigenvalues: 1D array of shape (k,)
        - eigenvectors: 2D array of shape (n, k), column i is eigenvector for eigenvalue i

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import compute_eigenpairs
    >>> G = nx.path_graph(5)
    >>> eigenvalues, eigenvectors = compute_eigenpairs(G, k=3)
    >>> eigenvalues.shape
    (3,)
    >>> eigenvectors.shape
    (5, 3)
    """
    n = G.number_of_nodes()

    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)

    # Use dense solver for small graphs, sparse for large
    if k >= n - 1 or n < 50:
        # Dense solver
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigenvectors = eigh(L_dense)
        # Take first k
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    else:
        # Sparse solver
        eigenvalues, eigenvectors = eigsh(L, k=k, which=which)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def algebraic_connectivity(G: nx.Graph, normalized: bool = False) -> float:
    """
    Compute the algebraic connectivity (Fiedler value) of a graph.

    The algebraic connectivity is the second smallest eigenvalue of the
    Laplacian. It is positive if and only if the graph is connected.
    Larger values indicate better connectivity.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    normalized : bool, default False
        If True, use normalized Laplacian

    Returns
    -------
    float
        The algebraic connectivity (λ₂)

    Raises
    ------
    ValueError
        If the graph is disconnected (λ₂ = 0)

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import algebraic_connectivity
    >>> G = nx.path_graph(5)
    >>> ac = algebraic_connectivity(G)
    >>> f"{ac:.4f}"
    '0.3820'
    """
    eigenvalues = compute_spectrum(G, k=2, normalized=normalized, which="SM")

    if len(eigenvalues) < 2:
        raise ValueError("Graph too small to compute algebraic connectivity")

    lambda2 = float(eigenvalues[1])

    if lambda2 < 1e-10:
        raise ValueError("Graph is disconnected (algebraic connectivity = 0)")

    return lambda2


if __name__ == "__main__":
    # Run simple validation
    import networkx as nx

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
