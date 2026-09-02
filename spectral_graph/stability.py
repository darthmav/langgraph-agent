"""
Numerical stability utilities for spectral graph operations.

Provides functions for:
- Condition number estimation
- Floating-point tolerance checking
- Eigenvalue stability validation
- Sparse matrix stability checks

All functions account for computational complexity and floating-point
precision limitations as per research findings.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from typing import Optional, Tuple


# Default floating-point tolerance
DEFAULT_TOL = 1e-10


def check_condition_number(
    matrix: np.ndarray,
    threshold: float = 1e10,
    tol: float = DEFAULT_TOL,
) -> Tuple[bool, float]:
    """
    Check if a matrix has acceptable condition number for stable operations.
    
    The condition number measures how sensitive a matrix is to numerical
    perturbations. High condition numbers indicate potential instability
    in eigenvalue computations and linear system solving.
    
    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse matrix
        Input matrix to check
    threshold : float, default 1e10
        Maximum acceptable condition number
    tol : float, default 1e-10
        Floating-point tolerance for singularity detection
    
    Returns
    -------
    tuple
        (is_stable, condition_number) where:
        - is_stable: True if condition number is below threshold
        - condition_number: estimated condition number (inf if singular)
    
    Notes
    -----
    Computational complexity: O(n³) for dense matrices using SVD.
    For large sparse matrices, consider using iterative estimators.
    
    Examples
    --------
    >>> import numpy as np
    >>> from spectral_graph.stability import check_condition_number
    >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> is_stable, cond = check_condition_number(A)
    >>> is_stable
    True
    >>> cond
    1.0
    """
    if sparse.issparse(matrix):
        # For sparse matrices, convert to dense for condition number
        # Note: This limits use to moderately sized sparse matrices
        matrix_dense = matrix.toarray()
    else:
        matrix_dense = np.asarray(matrix, dtype=np.float64)
    
    try:
        # Use SVD-based condition number for numerical stability
        singular_values = np.linalg.svd(matrix_dense, compute_uv=False)
        if len(singular_values) == 0:
            return True, 1.0
        
        sigma_max = np.max(singular_values)
        sigma_min = np.min(singular_values)
        
        if sigma_min < tol:
            return False, float('inf')
        
        cond = sigma_max / sigma_min
        is_stable = cond < threshold
        
        return is_stable, float(cond)
    
    except np.linalg.LinAlgError:
        return False, float('inf')


def check_eigenvalue_stability(
    eigenvalues: np.ndarray,
    tol: float = DEFAULT_TOL,
) -> Tuple[bool, dict]:
    """
    Validate eigenvalue computation for numerical stability issues.
    
    Checks for:
    - Negative eigenvalues in positive semi-definite matrices (Laplacians)
    - Duplicate eigenvalues (multiplicity detection)
    - Eigenvalue gaps (important for spectral clustering)
    - Floating-point precision artifacts
    
    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Computed eigenvalues (should be sorted in ascending order)
    tol : float, default 1e-10
        Floating-point tolerance for comparisons
    
    Returns
    -------
    tuple
        (is_stable, diagnostics) where:
        - is_stable: True if no significant stability issues detected
        - diagnostics: dict with keys:
            - 'has_negative': whether negative eigenvalues detected
            - 'min_eigenvalue': smallest eigenvalue
            - 'max_eigenvalue': largest eigenvalue
            - 'min_gap': smallest gap between consecutive eigenvalues
            - 'multiplicities': dict of eigenvalue -> multiplicity
    
    Examples
    --------
    >>> import numpy as np
    >>> from spectral_graph.stability import check_eigenvalue_stability
    >>> # Laplacian eigenvalues should be non-negative
    >>> evals = np.array([0.0, 0.5, 1.0, 2.0])
    >>> is_stable, diag = check_eigenvalue_stability(evals)
    >>> is_stable
    True
    >>> diag['has_negative']
    False
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    
    diagnostics = {
        'has_negative': False,
        'min_eigenvalue': float(np.min(eigenvalues)),
        'max_eigenvalue': float(np.max(eigenvalues)),
        'min_gap': float('inf'),
        'multiplicities': {},
    }
    
    # Check for negative eigenvalues (problematic for Laplacians)
    if np.any(eigenvalues < -tol):
        diagnostics['has_negative'] = True
    
    # Compute eigenvalue gaps
    if len(eigenvalues) > 1:
        gaps = np.diff(np.sort(eigenvalues))
        diagnostics['min_gap'] = float(np.min(gaps))
    
    # Detect multiplicities (eigenvalues within tolerance)
    sorted_evals = np.sort(eigenvalues)
    current_mult = 1
    for i in range(1, len(sorted_evals)):
        if abs(sorted_evals[i] - sorted_evals[i-1]) < tol:
            current_mult += 1
        else:
            key = float(sorted_evals[i-1])
            diagnostics['multiplicities'][key] = current_mult
            current_mult = 1
    # Don't forget the last group
    if len(sorted_evals) > 0:
        key = float(sorted_evals[-1])
        diagnostics['multiplicities'][key] = current_mult
    
    # Determine stability
    is_stable = not diagnostics['has_negative']
    
    return is_stable, diagnostics


def choose_eigen_solver(
    n_nodes: int,
    k_eigenpairs: int,
    is_sparse: bool = True,
) -> str:
    """
    Recommend eigenvalue solver based on problem size and sparsity.
    
    Based on research findings:
    - Use dense solver (numpy.linalg.eigh) for small graphs (<50 nodes)
    - Use sparse iterative solver (scipy.sparse.linalg.eigsh) for larger graphs
    - Lanczos/Arnoldi methods preferred for large sparse matrices
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph
    k_eigenpairs : int
        Number of eigenpairs needed
    is_sparse : bool, default True
        Whether the matrix is sparse
    
    Returns
    -------
    str
        Recommended solver: 'dense' or 'sparse'
    
    Examples
    --------
    >>> from spectral_graph.stability import choose_eigen_solver
    >>> choose_eigen_solver(30, 5)
    'dense'
    >>> choose_eigen_solver(1000, 10)
    'sparse'
    """
    # Threshold from research: use dense for n < 50
    DENSE_THRESHOLD = 50
    
    if n_nodes < DENSE_THRESHOLD:
        return 'dense'
    
    if is_sparse and n_nodes >= DENSE_THRESHOLD:
        return 'sparse'
    
    # For dense large matrices, still prefer sparse if possible
    # but warn about memory
    return 'sparse'


def verify_psd(
    matrix: np.ndarray,
    tol: float = DEFAULT_TOL,
) -> Tuple[bool, float]:
    """
    Verify that a matrix is positive semi-definite (PSD).
    
    Laplacian matrices should be PSD. This function checks by computing
    the minimum eigenvalue.
    
    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse matrix
        Input matrix to check
    tol : float, default 1e-10
        Tolerance for negative eigenvalue detection
    
    Returns
    -------
    tuple
        (is_psd, min_eigenvalue) where:
        - is_psd: True if matrix is PSD (all eigenvalues >= -tol)
        - min_eigenvalue: smallest eigenvalue found
    
    Notes
    -----
    Computational complexity: O(n³) for full eigenvalue decomposition.
    For large matrices, consider using eigsh with which='SA' to find
    only the smallest eigenvalue.
    
    Examples
    --------
    >>> import numpy as np
    >>> from spectral_graph.stability import verify_psd
    >>> # Identity matrix is PSD
    >>> is_psd, min_eval = verify_psd(np.eye(3))
    >>> is_psd
    True
    >>> abs(min_eval - 1.0) < 1e-10
    True
    """
    if sparse.issparse(matrix):
        # For sparse matrices, use sparse eigenvalue solver
        # to find just the smallest eigenvalue
        from scipy.sparse.linalg import eigsh
        try:
            min_eval = eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
        except Exception:
            # Fall back to dense for small matrices
            min_eval = np.linalg.eigvalsh(matrix.toarray())[0]
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        min_eval = np.linalg.eigvalsh(matrix)[0]
    
    is_psd = min_eval >= -tol
    
    return is_psd, float(min_eval)


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    tol: float = DEFAULT_TOL,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Perform element-wise division with protection against division by zero.
    
    Useful for computing D^(-1/2) in normalized Laplacian construction
    where some nodes may have degree zero (isolated nodes).
    
    Parameters
    ----------
    numerator : numpy.ndarray
        Numerator array
    denominator : numpy.ndarray
        Denominator array
    tol : float, default 1e-10
        Threshold below which values are considered zero
    fill_value : float, default 0.0
        Value to use when denominator is near zero
    
    Returns
    -------
    numpy.ndarray
        Result of division with safe handling of zeros
    
    Examples
    --------
    >>> import numpy as np
    >>> from spectral_graph.stability import safe_divide
    >>> num = np.array([1.0, 2.0, 3.0])
    >>> denom = np.array([1.0, 0.0, 2.0])
    >>> result = safe_divide(num, denom)
    >>> result
    array([1. , 0. , 1.5])
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    
    # Create mask for safe division
    safe_mask = np.abs(denominator) > tol
    
    result = np.full_like(numerator, fill_value, dtype=np.float64)
    result[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    
    return result


def safe_sqrt_inverse(
    values: np.ndarray,
    tol: float = DEFAULT_TOL,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Compute 1/sqrt(x) safely, handling zeros and negative values.
    
    Used in normalized Laplacian construction: D^(-1/2).
    
    Parameters
    ----------
    values : numpy.ndarray
        Input values (should be non-negative for meaningful results)
    tol : float, default 1e-10
        Threshold below which values are considered zero
    fill_value : float, default 0.0
        Value to use when input is near zero or negative
    
    Returns
    -------
    numpy.ndarray
        Array of 1/sqrt(values) with safe handling
    
    Examples
    --------
    >>> import numpy as np
    >>> from spectral_graph.stability import safe_sqrt_inverse
    >>> vals = np.array([1.0, 4.0, 0.0, 9.0])
    >>> result = safe_sqrt_inverse(vals)
    >>> result
    array([1. , 0.5, 0. , 0.333...])
    """
    values = np.asarray(values, dtype=np.float64)
    
    # Handle negative values (shouldn't occur for degrees, but be safe)
    safe_mask = values > tol
    
    result = np.full_like(values, fill_value, dtype=np.float64)
    result[safe_mask] = 1.0 / np.sqrt(values[safe_mask])
    
    return result


if __name__ == "__main__":
    # Run validation tests
    import numpy as np
    from scipy import sparse
    
    print("Testing stability.py...")
    
    # Test check_condition_number
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    is_stable, cond = check_condition_number(A)
    assert is_stable, "Identity matrix should be stable"
    assert cond == 1.0, f"Identity condition number should be 1.0, got {cond}"
    print("✓ check_condition_number works correctly")
    
    # Test check_eigenvalue_stability
    evals = np.array([0.0, 0.5, 1.0, 2.0])
    is_stable, diag = check_eigenvalue_stability(evals)
    assert is_stable, "Non-negative eigenvalues should be stable"
    assert not diag['has_negative'], "Should not detect negative eigenvalues"
    print("✓ check_eigenvalue_stability works correctly")
    
    # Test with negative eigenvalues
    evals_neg = np.array([-0.1, 0.5, 1.0])
    is_stable, diag = check_eigenvalue_stability(evals_neg)
    assert not is_stable, "Negative eigenvalues should be unstable"
    assert diag['has_negative'], "Should detect negative eigenvalues"
    print("✓ Negative eigenvalue detection works")
    
    # Test choose_eigen_solver
    assert choose_eigen_solver(30, 5) == 'dense', "Small graph should use dense"
    assert choose_eigen_solver(1000, 10) == 'sparse', "Large graph should use sparse"
    print("✓ choose_eigen_solver works correctly")
    
    # Test verify_psd
    is_psd, min_eval = verify_psd(np.eye(3))
    assert is_psd, "Identity matrix should be PSD"
    print("✓ verify_psd works correctly")
    
    # Test with non-PSD matrix
    non_psd = np.array([[-1.0, 0.0], [0.0, 1.0]])
    is_psd, min_eval = verify_psd(non_psd)
    assert not is_psd, "Matrix with negative eigenvalue should not be PSD"
    print("✓ PSD detection works correctly")
    
    # Test safe_divide
    num = np.array([1.0, 2.0, 3.0])
    denom = np.array([1.0, 0.0, 2.0])
    result = safe_divide(num, denom)
    expected = np.array([1.0, 0.0, 1.5])
    assert np.allclose(result, expected), f"safe_divide failed: {result} vs {expected}"
    print("✓ safe_divide works correctly")
    
    # Test safe_sqrt_inverse
    vals = np.array([1.0, 4.0, 0.0, 9.0])
    result = safe_sqrt_inverse(vals)
    expected = np.array([1.0, 0.5, 0.0, 1.0/3.0])
    assert np.allclose(result, expected), f"safe_sqrt_inverse failed: {result} vs {expected}"
    print("✓ safe_sqrt_inverse works correctly")
    
    # Test sparse matrix support
    sparse_mat = sparse.eye(100, format='csr')
    is_stable, cond = check_condition_number(sparse_mat)
    assert is_stable, "Sparse identity should be stable"
    print("✓ Sparse matrix support works")
    
    print("\nAll stability.py tests passed!")
