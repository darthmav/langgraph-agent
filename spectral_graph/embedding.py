"""
Spectral embedding via Laplacian eigenvectors.

Provides functions to compute low-dimensional embeddings of graphs using
the eigenvectors corresponding to the smallest non-zero eigenvalues of
the graph Laplacian.
"""

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from spectral_graph.laplacian import laplacian_matrix, normalized_laplacian_matrix
from spectral_graph.spectrum import smallest_eigsh


def spectral_embedding(
    G: nx.Graph,
    dim: int = 2,
    normalized: bool = False,
    use_fiedler: bool = True
) -> np.ndarray:
    """
    Compute spectral embedding of a graph.

    The embedding uses the eigenvectors corresponding to the smallest
    non-zero eigenvalues of the Laplacian. By default, it skips the first
    eigenvector (constant vector) and uses the next `dim` eigenvectors.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    dim : int, default 2
        Dimension of the embedding
    normalized : bool, default False
        If True, use normalized Laplacian; otherwise use unnormalized
    use_fiedler : bool, default True
        If True, skip the first eigenvector (constant) and start from Fiedler

    Returns
    -------
    numpy.ndarray
        Embedding matrix of shape (n_nodes, dim)

    Raises
    ------
    ValueError
        If dim is too large or graph has insufficient nodes

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import spectral_embedding
    >>> G = nx.path_graph(10)
    >>> embedding = spectral_embedding(G, dim=2)
    >>> embedding.shape
    (10, 2)
    """
    n = G.number_of_nodes()

    if n < dim + (1 if use_fiedler else 0):
        raise ValueError(f"Graph has {n} nodes, need at least {dim + (1 if use_fiedler else 0)} for {dim}D embedding")

    if dim < 1:
        raise ValueError("Embedding dimension must be at least 1")

    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)

    # Number of eigenpairs to compute
    k = dim + (1 if use_fiedler else 0)

    # Compute eigenpairs
    if n < 50:
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigenvectors = eigh(L_dense)
        # Take the k smallest
        idx = np.argsort(eigenvalues)[:k]
        eigenvectors = eigenvectors[:, idx]
    else:
        # Sparse solver, shift-inverted; see `smallest_eigsh`.
        eigenvalues, eigenvectors = smallest_eigsh(L, k=k)

    # Skip the first eigenvector if use_fiedler (it's the constant vector)
    if use_fiedler:
        embedding = eigenvectors[:, 1:dim+1]
    else:
        embedding = eigenvectors[:, :dim]

    return np.asarray(embedding)


def laplacian_eigenmap(
    G: nx.Graph,
    dim: int = 2,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute Laplacian Eigenmap embedding.

    This is a variant of spectral embedding that specifically uses the
    normalized Laplacian and skips the first eigenvector, following the
    original Laplacian Eigenmaps algorithm.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    dim : int, default 2
        Dimension of the embedding
    normalized : bool, default True
        Always use normalized Laplacian for Laplacian Eigenmaps

    Returns
    -------
    numpy.ndarray
        Embedding matrix of shape (n_nodes, dim)

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import laplacian_eigenmap
    >>> G = nx.karate_club_graph()
    >>> embedding = laplacian_eigenmap(G, dim=2)
    >>> embedding.shape
    (34, 2)
    """
    return spectral_embedding(G, dim=dim, normalized=normalized, use_fiedler=True)


def embed_and_normalize(
    G: nx.Graph,
    dim: int = 2,
    normalized: bool = False
) -> np.ndarray:
    """
    Compute spectral embedding and normalize rows to unit length.

    This is useful for clustering applications where the direction of
    the embedding vector matters more than its magnitude.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    dim : int, default 2
        Dimension of the embedding
    normalized : bool, default False
        If True, use normalized Laplacian

    Returns
    -------
    numpy.ndarray
        Row-normalized embedding matrix of shape (n_nodes, dim)

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import embed_and_normalize
    >>> G = nx.path_graph(10)
    >>> embedding = embed_and_normalize(G, dim=2)
    >>> # Each row should have unit norm
    >>> import numpy as np
    >>> norms = np.linalg.norm(embedding, axis=1)
    >>> np.allclose(norms, 1.0)
    True
    """
    embedding = spectral_embedding(G, dim=dim, normalized=normalized)

    # Normalize each row to unit length
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    # Avoid division by zero
    row_norms = np.where(row_norms > 1e-10, row_norms, 1.0)
    normalized_embedding = embedding / row_norms

    return normalized_embedding


if __name__ == "__main__":
    # This script should not be run directly from within spectral_graph/
    # Run via: python verify_embedding.py from project root
    print("Do not run embedding.py directly. Use verify_embedding.py from project root.")
