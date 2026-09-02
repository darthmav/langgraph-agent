"""
Spectral graph arithmetic operations.

This module implements core spectral graph arithmetic operations as defined
in the research findings:

1. Spectrum extraction with stability checks
2. Spectral filtering (operations on eigenvalue domain)
3. Laplacian arithmetic (matrix operations with sparsity preservation)

Computational complexity is documented in docstrings:
- Dense operations: O(n³) for full eigenvalue decomposition
- Sparse iterative: O(k·n) for k eigenpairs using Lanczos/Arnoldi methods

All operations use standard scientific Python libraries (NumPy, SciPy, NetworkX)
and account for numerical stability and floating-point precision limitations.
"""

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from typing import Callable, Optional, Tuple, Union

from spectral_graph.laplacian import (
    laplacian_matrix,
    normalized_laplacian_matrix,
    adjacency_matrix,
)
from spectral_graph.stability import (
    DEFAULT_TOL,
    check_eigenvalue_stability,
    choose_eigen_solver,
    safe_sqrt_inverse,
    verify_psd,
)


def compute_spectrum_stable(
    G: nx.Graph,
    k: Optional[int] = None,
    normalized: bool = False,
    tol: float = DEFAULT_TOL,
    check_stability: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Compute eigenvalues of the graph Laplacian with stability checks.
    
    Automatically chooses between dense and sparse solvers based on graph size.
    Includes optional stability diagnostics for numerical precision validation.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    k : int, optional
        Number of eigenvalues to compute. If None, computes all eigenvalues.
    normalized : bool, default False
        If True, use normalized Laplacian; otherwise use unnormalized.
    tol : float, default 1e-10
        Floating-point tolerance for stability checks
    check_stability : bool, default True
        If True, return stability diagnostics along with eigenvalues
    
    Returns
    -------
    numpy.ndarray or tuple
        If check_stability is False: array of eigenvalues in ascending order
        If check_stability is True: (eigenvalues, diagnostics) where diagnostics
        contains stability information from check_eigenvalue_stability
    
    Notes
    -----
    Computational complexity:
    - Dense solver (n < 50): O(n³) for full decomposition
    - Sparse solver (n >= 50): O(k·n·iter) for k eigenpairs using ARPACK
    
    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph.operations import compute_spectrum_stable
    >>> G = nx.path_graph(5)
    >>> eigenvalues, diag = compute_spectrum_stable(G, check_stability=True)
    >>> eigenvalues.shape
    (5,)
    >>> diag['has_negative']
    False
    """
    n = G.number_of_nodes()
    
    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)
    
    # Choose solver based on graph size
    solver = choose_eigen_solver(n, k if k else n, sparse.issparse(L))
    
    if solver == 'dense' or k is None or k >= n - 1:
        # Dense solver - compute all eigenvalues
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues = np.linalg.eigvalsh(L_dense)
    else:
        # Sparse solver - compute k eigenvalues
        eigenvalues = eigsh(L, k=k, which='SM', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
    
    if check_stability:
        is_stable, diagnostics = check_eigenvalue_stability(eigenvalues, tol=tol)
        diagnostics['is_stable'] = is_stable
        diagnostics['solver_used'] = solver
        return eigenvalues, diagnostics
    
    return eigenvalues


def spectral_filter(
    G: nx.Graph,
    signal: np.ndarray,
    filter_func: Callable[[np.ndarray], np.ndarray],
    k: Optional[int] = None,
    normalized: bool = False,
    tol: float = DEFAULT_TOL,
) -> np.ndarray:
    """
    Apply a spectral filter to a graph signal.
    
    Spectral filtering operates in the eigenvalue domain:
    1. Decompose signal into Laplacian eigenvector basis
    2. Apply filter function to eigenvalue coefficients
    3. Reconstruct filtered signal
    
    This is the foundation for spectral graph convolutions and
    graph signal processing operations.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    signal : numpy.ndarray
        Graph signal (1D array of length n_nodes)
    filter_func : callable
        Function that takes eigenvalues and returns filter coefficients.
        Should accept a 1D array and return a 1D array of same length.
        Example: lambda x: np.exp(-x) for low-pass filtering
    k : int, optional
        Number of eigenpairs to use. If None, uses all eigenpairs.
    normalized : bool, default False
        If True, use normalized Laplacian
    tol : float, default 1e-10
        Tolerance for numerical stability checks
    
    Returns
    -------
    numpy.ndarray
        Filtered graph signal
    
    Notes
    -----
    Computational complexity: O(n³) for dense, O(k·n·iter) for sparse
    
    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from spectral_graph.operations import spectral_filter
    >>> G = nx.path_graph(10)
    >>> signal = np.random.randn(10)
    >>> # Low-pass filter: attenuate high frequencies
    >>> lowpass = lambda x: np.exp(-x / 2.0)
    >>> filtered = spectral_filter(G, signal, lowpass)
    >>> filtered.shape
    (10,)
    """
    n = G.number_of_nodes()
    signal = np.asarray(signal, dtype=np.float64).flatten()
    
    if len(signal) != n:
        raise ValueError(f"Signal length {len(signal)} does not match graph size {n}")
    
    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)
    
    # Determine number of eigenpairs
    if k is None or k >= n:
        k = n
        use_dense = n < 50
    else:
        use_dense = n < 50
    
    # Compute eigenpairs
    if use_dense or k >= n - 1:
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigenvectors = eigh(L_dense)
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    else:
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
    # Project signal onto eigenvector basis
    coefficients = eigenvectors.T @ signal
    
    # Apply filter in spectral domain
    filter_coeffs = filter_func(eigenvalues)
    filtered_coeffs = coefficients * filter_coeffs
    
    # Reconstruct signal
    filtered_signal = eigenvectors @ filtered_coeffs
    
    return filtered_signal


def laplacian_add(
    L1: sparse.spmatrix,
    L2: sparse.spmatrix,
    preserve_sparsity: bool = True,
) -> sparse.spmatrix:
    """
    Add two Laplacian matrices while preserving sparsity structure.
    
    Parameters
    ----------
    L1 : scipy.sparse matrix
        First Laplacian matrix
    L2 : scipy.sparse matrix
        Second Laplacian matrix
    preserve_sparsity : bool, default True
        If True, return result in sparse format; otherwise convert to dense
    
    Returns
    -------
    scipy.sparse matrix or numpy.ndarray
        Sum of the two Laplacians
    
    Notes
    -----
    Computational complexity: O(nnz) where nnz is number of non-zero elements
    
    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph.operations import laplacian_add
    >>> from spectral_graph.laplacian import laplacian_matrix
    >>> G1 = nx.path_graph(5)
    >>> G2 = nx.cycle_graph(5)
    >>> L1 = laplacian_matrix(G1)
    >>> L2 = laplacian_matrix(G2)
    >>> L_sum = laplacian_add(L1, L2)
    >>> L_sum.shape
    (5, 5)
    """
    if not sparse.issparse(L1):
        L1 = sparse.csr_matrix(L1)
    if not sparse.issparse(L2):
        L2 = sparse.csr_matrix(L2)
    
    result = L1 + L2
    
    if preserve_sparsity:
        return result.tocsr()
    else:
        return result.toarray()


def laplacian_scale(
    L: Union[sparse.spmatrix, np.ndarray],
    scalar: float,
    preserve_sparsity: bool = True,
) -> Union[sparse.spmatrix, np.ndarray]:
    """
    Scale a Laplacian matrix by a scalar.
    
    Parameters
    ----------
    L : scipy.sparse matrix or numpy.ndarray
        Laplacian matrix
    scalar : float
        Scaling factor
    preserve_sparsity : bool, default True
        If True and input is sparse, return sparse result
    
    Returns
    -------
    scipy.sparse matrix or numpy.ndarray
        Scaled Laplacian
    
    Notes
    -----
    Computational complexity: O(nnz) for sparse, O(n²) for dense
    
    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph.operations import laplacian_scale
    >>> from spectral_graph.laplacian import laplacian_matrix
    >>> G = nx.path_graph(5)
    >>> L = laplacian_matrix(G)
    >>> L_scaled = laplacian_scale(L, 2.0)
    >>> # Verify scaling
    >>> import numpy as np
    >>> np.allclose(L_scaled.toarray(), 2.0 * L.toarray())
    True
    """
    if sparse.issparse(L):
        result = L.multiply(scalar)
        if preserve_sparsity:
            return result.tocsr()
        else:
            return result.toarray()
    else:
        return np.asarray(L, dtype=np.float64) * scalar


def laplacian_convex_combination(
    L1: sparse.spmatrix,
    L2: sparse.spmatrix,
    alpha: float,
    preserve_sparsity: bool = True,
) -> sparse.spmatrix:
    """
    Compute convex combination of two Laplacians: α·L1 + (1-α)·L2.
    
    Useful for interpolating between different graph structures or
    combining multiple graph Laplacians.
    
    Parameters
    ----------
    L1 : scipy.sparse matrix
        First Laplacian matrix
    L2 : scipy.sparse matrix
        Second Laplacian matrix
    alpha : float
        Weight for L1 (should be in [0, 1] for convex combination)
    preserve_sparsity : bool, default True
        If True, return result in sparse format
    
    Returns
    -------
    scipy.sparse matrix
        Convex combination of Laplacians
    
    Raises
    ------
    ValueError
        If alpha is outside [0, 1] range
    
    Notes
    -----
    Computational complexity: O(nnz) where nnz is number of non-zero elements
    
    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph.operations import laplacian_convex_combination
    >>> from spectral_graph.laplacian import laplacian_matrix
    >>> G1 = nx.path_graph(5)
    >>> G2 = nx.cycle_graph(5)
    >>> L1 = laplacian_matrix(G1)
    >>> L2 = laplacian_matrix(G2)
    >>> L_combined = laplacian_convex_combination(L1, L2, 0.5)
    >>> L_combined.shape
    (5, 5)
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
    
    if not sparse.issparse(L1):
        L1 = sparse.csr_matrix(L1)
    if not sparse.issparse(L2):
        L2 = sparse.csr_matrix(L2)
    
    result = L1.multiply(alpha) + L2.multiply(1.0 - alpha)
    
    if preserve_sparsity:
        return result.tocsr()
    else:
        return result.toarray()


def normalized_laplacian_from_unnormalized(
    L: sparse.spmatrix,
    degrees: Optional[np.ndarray] = None,
    tol: float = DEFAULT_TOL,
) -> sparse.spmatrix:
    """
    Compute normalized Laplacian from unnormalized Laplacian.
    
    L_norm = I - D^(-1/2) A D^(-1/2) = D^(-1/2) L D^(-1/2)
    
    Parameters
    ----------
    L : scipy.sparse matrix
        Unnormalized Laplacian matrix
    degrees : numpy.ndarray, optional
        Pre-computed degree vector. If None, extracts from L diagonal.
    tol : float, default 1e-10
        Tolerance for zero-degree detection
    
    Returns
    -------
    scipy.sparse matrix
        Normalized Laplacian matrix
    
    Notes
    -----
    Computational complexity: O(nnz) for sparse matrix operations
    
    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph.operations import normalized_laplacian_from_unnormalized
    >>> from spectral_graph.laplacian import laplacian_matrix
    >>> G = nx.path_graph(5)
    >>> L = laplacian_matrix(G)
    >>> L_norm = normalized_laplacian_from_unnormalized(L)
    >>> L_norm.shape
    (5, 5)
    """
    n = L.shape[0]
    
    if degrees is None:
        # Extract degrees from diagonal of L
        degrees = np.asarray(L.diagonal(), dtype=np.float64)
    
    # Compute D^(-1/2)
    d_inv_sqrt = safe_sqrt_inverse(degrees, tol=tol, fill_value=0.0)
    D_inv_sqrt = sparse.diags(d_inv_sqrt, format='csr')
    
    # L_norm = D^(-1/2) L D^(-1/2)
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    return L_norm.tocsr()


def spectral_distance(
    G1: nx.Graph,
    G2: nx.Graph,
    k: Optional[int] = None,
    normalized: bool = True,
    distance_type: str = 'euclidean',
) -> float:
    """
    Compute distance between two graphs based on their spectra.
    
    Spectral distance measures how similar two graphs are by comparing
    their Laplacian eigenvalues. Useful for graph comparison and clustering.
    
    Parameters
    ----------
    G1, G2 : networkx.Graph
        Input graphs to compare
    k : int, optional
        Number of eigenvalues to compare. If None, uses min(n1, n2).
    normalized : bool, default True
        If True, use normalized Laplacians for comparison
    distance_type : str, default 'euclidean'
        Type of distance metric:
        - 'euclidean': L2 distance between eigenvalue vectors
        - 'cosine': cosine distance
        - 'wasserstein': 1D Wasserstein distance (requires sorted eigenvalues)
    
    Returns
    -------
    float
        Distance between the two graphs
    
    Raises
    ------
    ValueError
        If graphs have incompatible sizes or invalid distance_type
    
    Notes
    -----
    Computational complexity: O(n³) for eigenvalue computation
    
    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph.operations import spectral_distance
    >>> G1 = nx.path_graph(5)
    >>> G2 = nx.path_graph(5)
    >>> dist = spectral_distance(G1, G2)
    >>> abs(dist) < 1e-10  # Same graph structure
    True
    """
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    
    if k is None:
        k = min(n1, n2)
    
    if k > n1 or k > n2:
        raise ValueError(f"Cannot compare {k} eigenvalues when graphs have {n1} and {n2} nodes")
    
    # Compute spectra
    evals1 = compute_spectrum_stable(G1, k=k, normalized=normalized, check_stability=False)
    evals2 = compute_spectrum_stable(G2, k=k, normalized=normalized, check_stability=False)
    
    # Ensure same length (pad with zeros if needed)
    if len(evals1) < len(evals2):
        evals1 = np.pad(evals1, (0, len(evals2) - len(evals1)))
    elif len(evals2) < len(evals1):
        evals2 = np.pad(evals2, (0, len(evals1) - len(evals2)))
    
    if distance_type == 'euclidean':
        return float(np.linalg.norm(evals1 - evals2))
    elif distance_type == 'cosine':
        norm1 = np.linalg.norm(evals1)
        norm2 = np.linalg.norm(evals2)
        if norm1 < DEFAULT_TOL or norm2 < DEFAULT_TOL:
            return 0.0 if np.allclose(evals1, evals2) else 1.0
        cosine_sim = np.dot(evals1, evals2) / (norm1 * norm2)
        return float(1.0 - cosine_sim)
    elif distance_type == 'wasserstein':
        # For 1D distributions, Wasserstein-1 is integral of |CDF1 - CDF2|
        # For sorted eigenvalues, this simplifies to L1 distance
        return float(np.sum(np.abs(evals1 - evals2)))
    else:
        raise ValueError(f"Unknown distance_type: {distance_type}")


def cheeger_constant_estimate(
    G: nx.Graph,
    normalized: bool = True,
) -> Tuple[float, float]:
    """
    Estimate Cheeger constant using spectral bounds.
    
    The Cheeger inequality provides bounds on the conductance h(G):
        λ₂/2 ≤ h(G) ≤ √(2λ₂)
    
    where λ₂ is the second smallest eigenvalue (algebraic connectivity).
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    normalized : bool, default True
        If True, use normalized Laplacian (standard for Cheeger bounds)
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound) on the Cheeger constant
    
    Raises
    ------
    ValueError
        If graph is too small
    
    Notes
    -----
    Computational complexity: O(n³) dense, O(k·n·iter) sparse for eigenvalue
    
    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph.operations import cheeger_constant_estimate
    >>> G = nx.path_graph(10)
    >>> lower, upper = cheeger_constant_estimate(G)
    >>> lower <= upper
    True
    """
    n = G.number_of_nodes()
    if n < 2:
        raise ValueError("Graph must have at least 2 nodes")
    
    # Compute second eigenvalue
    eigenvalues = compute_spectrum_stable(
        G, k=2, normalized=normalized, check_stability=False
    )
    
    if len(eigenvalues) < 2:
        raise ValueError("Could not compute enough eigenvalues")
    
    lambda2 = float(eigenvalues[1])
    lambda2 = max(lambda2, 0.0)  # Clamp numerical noise
    
    lower_bound = lambda2 / 2.0
    upper_bound = np.sqrt(2.0 * lambda2)
    
    return lower_bound, upper_bound


if __name__ == "__main__":
    # Run validation tests
    import networkx as nx
    
    print("Testing operations.py...")
    
    # Test compute_spectrum_stable
    G = nx.path_graph(5)
    eigenvalues, diag = compute_spectrum_stable(G, check_stability=True)
    assert len(eigenvalues) == 5, f"Expected 5 eigenvalues, got {len(eigenvalues)}"
    assert diag['is_stable'], "Path graph spectrum should be stable"
    assert not diag['has_negative'], "Laplacian eigenvalues should be non-negative"
    print("✓ compute_spectrum_stable works correctly")
    
    # Test spectral_filter
    G = nx.path_graph(10)
    signal = np.random.randn(10)
    lowpass = lambda x: np.exp(-x / 2.0)
    filtered = spectral_filter(G, signal, lowpass)
    assert filtered.shape == (10,), f"Filtered signal shape mismatch: {filtered.shape}"
    print("✓ spectral_filter works correctly")
    
    # Test laplacian_add
    G1 = nx.path_graph(5)
    G2 = nx.cycle_graph(5)
    L1 = laplacian_matrix(G1)
    L2 = laplacian_matrix(G2)
    L_sum = laplacian_add(L1, L2)
    assert L_sum.shape == (5, 5), "Laplacian sum shape mismatch"
    expected = L1.toarray() + L2.toarray()
    assert np.allclose(L_sum.toarray(), expected), "Laplacian addition failed"
    print("✓ laplacian_add works correctly")
    
    # Test laplacian_scale
    L_scaled = laplacian_scale(L1, 2.0)
    assert np.allclose(L_scaled.toarray(), 2.0 * L1.toarray()), "Laplacian scaling failed"
    print("✓ laplacian_scale works correctly")
    
    # Test laplacian_convex_combination
    L_combined = laplacian_convex_combination(L1, L2, 0.5)
    expected = 0.5 * L1.toarray() + 0.5 * L2.toarray()
    assert np.allclose(L_combined.toarray(), expected), "Convex combination failed"
    print("✓ laplacian_convex_combination works correctly")
    
    # Test normalized_laplacian_from_unnormalized
    L_norm = normalized_laplacian_from_unnormalized(L1)
    from spectral_graph.laplacian import normalized_laplacian_matrix
    L_norm_expected = normalized_laplacian_matrix(G1)
    assert np.allclose(L_norm.toarray(), L_norm_expected.toarray()), \
        "Normalized Laplacian conversion failed"
    print("✓ normalized_laplacian_from_unnormalized works correctly")
    
    # Test spectral_distance
    G1 = nx.path_graph(5)
    G2 = nx.path_graph(5)
    dist = spectral_distance(G1, G2)
    assert abs(dist) < 1e-10, f"Identical graphs should have zero distance, got {dist}"
    print("✓ spectral_distance works correctly")
    
    # Test with different graphs
    G3 = nx.cycle_graph(5)
    dist_diff = spectral_distance(G1, G3)
    assert dist_diff > 0, "Different graphs should have non-zero distance"
    print("✓ spectral_distance distinguishes different graphs")
    
    # Test cheeger_constant_estimate
    G = nx.path_graph(10)
    lower, upper = cheeger_constant_estimate(G)
    assert lower <= upper, "Cheeger bounds should satisfy lower <= upper"
    assert lower >= 0, "Lower bound should be non-negative"
    print("✓ cheeger_constant_estimate works correctly")
    
    # Test with larger graph (sparse solver)
    G_large = nx.barabasi_albert_graph(100, 3, seed=42)
    eigenvalues, diag = compute_spectrum_stable(G_large, k=10, check_stability=True)
    assert len(eigenvalues) == 10, f"Expected 10 eigenvalues, got {len(eigenvalues)}"
    assert diag['solver_used'] == 'sparse', "Large graph should use sparse solver"
    print("✓ Sparse solver works for larger graphs")
    
    print("\nAll operations.py tests passed!")
