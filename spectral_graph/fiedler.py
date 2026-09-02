"""
Fiedler vector and spectral bipartitioning.

Provides functions to compute the Fiedler vector (eigenvector corresponding
to the second smallest eigenvalue of the Laplacian) and use it for graph
bipartitioning.
"""

from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from spectral_graph.laplacian import (
    _require_undirected,
    laplacian_matrix,
    normalized_laplacian_matrix,
)
from spectral_graph.spectrum import smallest_eigsh


def fiedler_vector(G: nx.Graph, normalized: bool = False) -> np.ndarray:
    """
    Compute the Fiedler vector of a graph.

    The Fiedler vector is the eigenvector corresponding to the second
    smallest eigenvalue of the Laplacian (the algebraic connectivity).
    It is used for spectral bipartitioning.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (must be connected)
    normalized : bool, default False
        If True, use normalized Laplacian; otherwise use unnormalized.

    Returns
    -------
    numpy.ndarray
        The Fiedler vector (1D array of length n)

    Raises
    ------
    ValueError
        If the graph is disconnected or has fewer than 2 nodes

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import fiedler_vector
    >>> G = nx.path_graph(5)
    >>> fiedler = fiedler_vector(G)
    >>> fiedler.shape
    (5,)
    >>> # Fiedler vector should be orthogonal to constant vector
    >>> import numpy as np
    >>> np.abs(np.sum(fiedler)) < 1e-10
    True
    """
    # Ahead of nx.is_connected, which raises NetworkXNotImplemented on a
    # directed graph -- a true refusal, but one that says nothing about why
    # this package will not take it or what to do instead.
    _require_undirected(G)

    n = G.number_of_nodes()

    if n < 2:
        raise ValueError("Graph must have at least 2 nodes")

    # Check connectivity
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected to compute Fiedler vector")

    # Choose Laplacian type
    if normalized:
        L = normalized_laplacian_matrix(G)
    else:
        L = laplacian_matrix(G)

    # Compute the first 2 eigenpairs
    # Use dense solver for small graphs
    if n < 50:
        L_dense = L.toarray() if sparse.issparse(L) else L
        eigenvalues, eigenvectors = eigh(L_dense)
        fiedler = eigenvectors[:, 1]  # Second eigenvector (index 1)
    else:
        # Sparse solver for k=2 smallest eigenvalues; see `smallest_eigsh` for
        # why this is shift-inverted rather than `which="SM"`.
        eigenvalues, eigenvectors = smallest_eigsh(L, k=2)
        fiedler = eigenvectors[:, 1]

    return np.asarray(fiedler)


def fiedler_partition(G: nx.Graph, normalized: bool = False) -> tuple[set[Any], set[Any]]:
    """
    Partition a graph using the Fiedler vector.

    Nodes are partitioned into two sets based on the sign of their
    corresponding entry in the Fiedler vector.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (must be connected)
    normalized : bool, default False
        If True, use normalized Laplacian

    Returns
    -------
    tuple
        (set1, set2) where each set contains node indices

    Raises
    ------
    ValueError
        If the graph is disconnected or has fewer than 2 nodes

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import fiedler_partition
    >>> G = nx.path_graph(5)
    >>> set1, set2 = fiedler_partition(G)
    >>> len(set1) + len(set2) == 5
    True
    """
    fiedler = fiedler_vector(G, normalized=normalized)
    nodes = list(G.nodes())

    # Partition based on sign of Fiedler vector entries
    set1 = {nodes[i] for i in range(len(nodes)) if fiedler[i] >= 0}
    set2 = {nodes[i] for i in range(len(nodes)) if fiedler[i] < 0}

    return set1, set2


def spectral_bipartition(G: nx.Graph, normalized: bool = False) -> dict[str, Any]:
    """
    Perform spectral bipartitioning on a graph.

    Returns detailed information about the bipartition including
    the cut size and partition quality metrics.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (must be connected)
    normalized : bool, default False
        If True, use normalized Laplacian

    Returns
    -------
    dict
        Dictionary containing:
        - 'set1': set of nodes in partition 1
        - 'set2': set of nodes in partition 2
        - 'cut_size': number of edges crossing the cut
        - 'balance': ratio of smaller partition size to larger
        - 'fiedler_value': the algebraic connectivity

    Raises
    ------
    ValueError
        If the graph is disconnected or has fewer than 2 nodes

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import spectral_bipartition
    >>> G = nx.path_graph(5)
    >>> result = spectral_bipartition(G)
    >>> 'set1' in result and 'set2' in result
    True
    >>> result['cut_size']
    1
    """
    set1, set2 = fiedler_partition(G, normalized=normalized)

    # Compute cut size
    cut_size = 0
    for u, v in G.edges():
        if (u in set1 and v in set2) or (u in set2 and v in set1):
            cut_size += 1

    # Compute balance
    n = G.number_of_nodes()
    balance = min(len(set1), len(set2)) / max(len(set1), len(set2))

    # Get Fiedler value
    from spectral_graph.spectrum import algebraic_connectivity
    fiedler_value = algebraic_connectivity(G, normalized=normalized)

    return {
        "set1": set1,
        "set2": set2,
        "cut_size": cut_size,
        "balance": balance,
        "fiedler_value": fiedler_value,
    }


if __name__ == "__main__":
    # This script should not be run directly from within spectral_graph/
    # Run via: python verify_fiedler.py from project root
    print("Do not run fiedler.py directly. Use verify_fiedler.py from project root.")
