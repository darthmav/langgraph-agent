"""
spectral_graph: A lightweight package for spectral graph theory implementations.

This package provides direct implementations of core spectral graph theory
algorithms using NumPy/SciPy, with NetworkX used only for graph construction
and validation.

Core functionality:
- Laplacian construction (unnormalized, normalized, random walk)
- Spectrum computation (eigenvalues and eigenvectors)
- Fiedler vector and algebraic connectivity
- Spectral clustering and embedding
- Cheeger constant and bounds

Example usage:
    import networkx as nx
    from spectral_graph import laplacian_matrix, fiedler_vector, spectral_clustering

    G = nx.karate_club_graph()
    L = laplacian_matrix(G)
    fiedler_val, fiedler_vec = fiedler_vector(G)
    labels = spectral_clustering(G, k=2)
"""

from spectral_graph.clustering import (
    cheeger_bounds,
    conductance,
    spectral_clustering,
    sweep_cut,
)
from spectral_graph.embedding import (
    embed_and_normalize,
    laplacian_eigenmap,
    spectral_embedding,
)
from spectral_graph.fiedler import (
    fiedler_partition,
    fiedler_vector,
    spectral_bipartition,
)
from spectral_graph.laplacian import (
    adjacency_matrix,
    degree_matrix,
    laplacian_matrix,
    normalized_laplacian_matrix,
    random_walk_laplacian_matrix,
)
from spectral_graph.spectrum import (
    algebraic_connectivity,
    compute_eigenpairs,
    compute_spectrum,
)

__version__ = "0.1.0"
__all__ = [
    # Laplacian construction
    "laplacian_matrix",
    "normalized_laplacian_matrix",
    "random_walk_laplacian_matrix",
    "degree_matrix",
    "adjacency_matrix",
    # Spectrum computation
    "compute_spectrum",
    "compute_eigenpairs",
    "algebraic_connectivity",
    # Fiedler vector
    "fiedler_vector",
    "fiedler_partition",
    "spectral_bipartition",
    # Embedding
    "spectral_embedding",
    "laplacian_eigenmap",
    "embed_and_normalize",
    # Clustering and conductance
    "spectral_clustering",
    "conductance",
    "sweep_cut",
    "cheeger_bounds",
]
