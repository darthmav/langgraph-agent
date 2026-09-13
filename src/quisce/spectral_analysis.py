"""
Spectral Graph Analysis Module for QuICSE.

This module provides spectral graph analysis capabilities that integrate with
the QuICSE (Quantum-Infused Cognitive Synthesis Engine) system. It consumes
the quiesce state representation and outputs spectral properties without
modifying the core quiesce convergence logic.

The module exposes a clean interface for the main execution loop to invoke
spectral analysis as a post-quiescence step or as an optional diagnostic.

All operations use established numerical linear algebra libraries (NumPy, SciPy,
NetworkX) already available in the project's ecosystem.

Example usage:
    import numpy as np
    import networkx as nx
    from quisce.spectral_analysis import SpectralAnalyzer, SpectralProperties

    # Create a sample graph representing system state
    G = nx.karate_club_graph()

    # Analyze spectrally
    analyzer = SpectralAnalyzer.from_graph(G)
    properties = analyzer.analyze()

    print(f"Algebraic connectivity: {properties.algebraic_connectivity}")
    print(f"Fiedler vector norm: {np.linalg.norm(properties.fiedler_vector)}")

    # Or analyze from adjacency matrix directly
    adj_matrix = nx.adjacency_matrix(G)
    analyzer2 = SpectralAnalyzer.from_adjacency_matrix(adj_matrix)
    properties2 = analyzer2.analyze(k=5)  # Get top 5 eigenpairs
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse

# Add project root to path for spectral_graph import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import spectral_graph utilities - these are already in the project
try:
    from spectral_graph import (
        compute_eigenpairs,
        compute_spectrum,
        laplacian_matrix,
        normalized_laplacian_matrix,
        spectral_clustering,
        sweep_cut,
    )
    SPECTRAL_GRAPH_AVAILABLE = True
except ImportError:
    SPECTRAL_GRAPH_AVAILABLE = False


@dataclass
class SpectralProperties:
    """
    Container for spectral graph analysis results.

    This dataclass holds the output of spectral analysis without causing
    side effects on any global solver state. All fields are immutable once set.

    Attributes:
        eigenvalues: Sorted eigenvalues of the Laplacian matrix
        eigenvectors: Corresponding eigenvectors (columns match eigenvalues order)
        algebraic_connectivity: Second smallest eigenvalue (lambda_2), or None if unavailable
        fiedler_vector: Eigenvector corresponding to algebraic connectivity, or None
        spectral_gap: Gap between first two non-zero eigenvalues, or None
        cheeger_constant: Approximate Cheeger constant from sweep cut, or None
        conductance: Conductance of the best sweep cut, or None
        num_components: Estimated number of connected components (from zero eigenvalues)
        is_connected: Whether the graph appears connected (lambda_2 > threshold)
        clustering: Optional cluster assignments if clustering was performed
        metadata: Additional diagnostic information
    """
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    algebraic_connectivity: float | None = None
    fiedler_vector: np.ndarray | None = None
    spectral_gap: float | None = None
    cheeger_constant: float | None = None
    conductance: float | None = None
    num_components: int | None = None
    is_connected: bool | None = None
    clustering: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of spectral properties."""
        lines = ["Spectral Properties Summary", "=" * 26]

        if self.eigenvalues is not None:
            lines.append(f"Eigenvalues computed: {len(self.eigenvalues)}")
            if len(self.eigenvalues) > 0:
                lines.append(f"  Min: {self.eigenvalues[0]:.6f}")
                lines.append(f"  Max: {self.eigenvalues[-1]:.6f}")

        if self.algebraic_connectivity is not None:
            lines.append(f"Algebraic connectivity (λ₂): {self.algebraic_connectivity:.6f}")

        if self.spectral_gap is not None:
            lines.append(f"Spectral gap: {self.spectral_gap:.6f}")

        if self.num_components is not None:
            lines.append(f"Estimated components: {self.num_components}")

        if self.is_connected is not None:
            status = "Connected" if self.is_connected else "Disconnected"
            lines.append(f"Connectivity status: {status}")

        if self.conductance is not None:
            lines.append(f"Best conductance: {self.conductance:.6f}")

        if self.clustering is not None:
            unique_labels = len(np.unique(self.clustering))
            lines.append(f"Clustering: {unique_labels} clusters")

        return "\n".join(lines)


class SpectralAnalyzer:
    """
    Spectral graph analyzer for QuICSE system state.

    This class provides a clean interface for computing spectral properties
    of graphs representing system state. It accepts graph or matrix input
    and returns results without modifying any global state.

    The analyzer can be invoked as a post-quiescence diagnostic step or
    as part of the main execution loop.

    Parameters
    ----------
    graph : networkx.Graph, optional
        Input graph for analysis. Must be undirected.
    laplacian : scipy.sparse matrix, optional
        Pre-computed Laplacian matrix. If provided, graph is not needed.
    normalized : bool, default False
        If True, use normalized Laplacian for analysis.

    Examples
    --------
    >>> import networkx as nx
    >>> from quisce.spectral_analysis import SpectralAnalyzer
    >>>
    >>> G = nx.path_graph(10)
    >>> analyzer = SpectralAnalyzer.from_graph(G)
    >>> properties = analyzer.analyze()
    >>> print(properties.algebraic_connectivity)  # doctest: +SKIP
    """

    def __init__(
        self,
        graph: nx.Graph | None = None,
        laplacian: sparse.spmatrix | None = None,
        normalized: bool = False,
    ):
        """
        Initialize the spectral analyzer.

        Parameters
        ----------
        graph : networkx.Graph, optional
            Input graph for analysis
        laplacian : scipy.sparse matrix, optional
            Pre-computed Laplacian matrix
        normalized : bool, default False
            If True, use normalized Laplacian
        """
        if graph is None and laplacian is None:
            raise ValueError("Either graph or laplacian must be provided")

        self._graph = graph
        self._laplacian = laplacian
        self._normalized = normalized
        self._n_nodes: int | None = None

        # Determine number of nodes
        if graph is not None:
            self._n_nodes = graph.number_of_nodes()
        elif laplacian is not None:
            self._n_nodes = laplacian.shape[0]

    @classmethod
    def from_graph(cls, graph: nx.Graph, normalized: bool = False) -> SpectralAnalyzer:
        """
        Create analyzer from a NetworkX graph.

        Parameters
        ----------
        graph : networkx.Graph
            Input graph. Should be undirected; directed graphs will be converted.
        normalized : bool, default False
            If True, use normalized Laplacian

        Returns
        -------
        SpectralAnalyzer
            Configured analyzer instance

        Raises
        ------
        ValueError
            If graph is empty
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot analyze empty graph")

        # Ensure undirected - spectral analysis requires symmetric matrices
        if graph.is_directed():
            graph = graph.to_undirected(as_view=True)

        return cls(graph=graph, normalized=normalized)

    @classmethod
    def from_adjacency_matrix(
        cls,
        adj_matrix: sparse.spmatrix | np.ndarray,
        normalized: bool = False,
    ) -> SpectralAnalyzer:
        """
        Create analyzer from an adjacency matrix.

        Parameters
        ----------
        adj_matrix : scipy.sparse matrix or numpy.ndarray
            Adjacency matrix of the graph
        normalized : bool, default False
            If True, use normalized Laplacian

        Returns
        -------
        SpectralAnalyzer
            Configured analyzer instance
        """
        # Convert to sparse if dense
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = sparse.csr_matrix(adj_matrix, dtype=np.float64)

        # Compute Laplacian: L = D - A
        degree = np.asarray(adj_matrix.sum(axis=1)).flatten().astype(np.float64)
        D = sparse.diags(degree, format='csr')
        laplacian = D - adj_matrix

        return cls(laplacian=laplacian, normalized=normalized)

    @classmethod
    def from_laplacian_matrix(
        cls,
        laplacian: sparse.spmatrix | np.ndarray,
        normalized: bool = False,
    ) -> SpectralAnalyzer:
        """
        Create analyzer from a pre-computed Laplacian matrix.

        Parameters
        ----------
        laplacian : scipy.sparse matrix or numpy.ndarray
            Laplacian matrix of the graph
        normalized : bool, default False
            If True, treat as normalized Laplacian

        Returns
        -------
        SpectralAnalyzer
            Configured analyzer instance
        """
        if isinstance(laplacian, np.ndarray):
            laplacian = sparse.csr_matrix(laplacian)

        return cls(laplacian=laplacian, normalized=normalized)

    def _get_laplacian(self) -> sparse.spmatrix:
        """Get or compute the Laplacian matrix."""
        if self._laplacian is not None:
            return self._laplacian

        if self._graph is None:
            raise ValueError("No graph or Laplacian available")

        if self._normalized:
            return normalized_laplacian_matrix(self._graph)
        else:
            return laplacian_matrix(self._graph)

    def analyze(
        self,
        k: int | None = None,
        compute_fiedler: bool = True,
        compute_clustering: bool = False,
        n_clusters: int = 2,
        tolerance: float = 1e-10,
    ) -> SpectralProperties:
        """
        Perform spectral analysis and return properties.

        This is the main entry point for spectral analysis. It computes
        eigenvalues, eigenvectors, and derived properties without modifying
        any global state.

        Parameters
        ----------
        k : int, optional
            Number of eigenpairs to compute. If None, computes all eigenvalues
            for small graphs (< 50 nodes) or 10 for larger graphs.
        compute_fiedler : bool, default True
            If True, compute the Fiedler vector and algebraic connectivity
        compute_clustering : bool, default False
            If True, perform spectral clustering
        n_clusters : int, default 2
            Number of clusters for spectral clustering
        tolerance : float, default 1e-10
            Tolerance for determining zero eigenvalues (for component count)

        Returns
        -------
        SpectralProperties
            Container with all computed spectral properties

        Raises
        ------
        RuntimeError
            If spectral_graph package is not available
        ValueError
            If graph is too small for requested analysis
        """
        if not SPECTRAL_GRAPH_AVAILABLE:
            raise RuntimeError(
                "spectral_graph package is not available. "
                "Ensure it is on the Python path."
            )

        n = self._n_nodes
        if n is None or n == 0:
            raise ValueError("Cannot analyze empty graph")

        # Default k based on graph size
        if k is None:
            k = min(10, max(2, n - 1))

        # Ensure k is valid
        k = min(k, n)

        properties = SpectralProperties()

        # Compute eigenpairs
        try:
            eigenvalues, eigenvectors = compute_eigenpairs(
                self._graph if self._graph is not None else self._create_graph_from_laplacian(),
                k=k,
                normalized=self._normalized,
            )
            properties.eigenvalues = eigenvalues
            properties.eigenvectors = eigenvectors
        except Exception as e:
            properties.metadata["eigen_computation_error"] = str(e)
            # Fall back to eigenvalues only
            try:
                eigenvalues = compute_spectrum(
                    self._graph if self._graph is not None else self._create_graph_from_laplacian(),
                    k=k,
                    normalized=self._normalized,
                )
                properties.eigenvalues = eigenvalues
            except Exception as e2:
                properties.metadata["spectrum_computation_error"] = str(e2)
                return properties

        # Count connected components from zero eigenvalues
        if properties.eigenvalues is not None:
            num_zeros = int(np.sum(np.abs(properties.eigenvalues) < tolerance))
            properties.num_components = max(num_zeros, 1)
            properties.is_connected = properties.num_components == 1

        # Compute algebraic connectivity and Fiedler vector
        if compute_fiedler and properties.eigenvalues is not None and len(properties.eigenvalues) >= 2:
            try:
                # Algebraic connectivity is the second smallest eigenvalue
                lambda2 = float(properties.eigenvalues[1])
                properties.algebraic_connectivity = lambda2

                # Spectral gap (difference between λ₂ and λ₃ if available)
                if len(properties.eigenvalues) >= 3:
                    properties.spectral_gap = float(properties.eigenvalues[2] - properties.eigenvalues[1])

                # Fiedler vector
                if properties.eigenvectors is not None and properties.eigenvectors.shape[1] >= 2:
                    properties.fiedler_vector = properties.eigenvectors[:, 1].copy()
            except Exception as e:
                properties.metadata["fiedler_computation_error"] = str(e)

        # Compute conductance via sweep cut
        if self._graph is not None and properties.fiedler_vector is not None:
            try:
                # Use sweep cut to find best conductance
                _, conductance_val = sweep_cut(self._graph)
                properties.conductance = float(conductance_val)

                # Cheeger constant approximation
                if properties.algebraic_connectivity is not None:
                    # Cheeger inequality: λ₂/2 ≤ h ≤ √(2λ₂)
                    properties.cheeger_constant = float(
                        np.sqrt(2 * properties.algebraic_connectivity)
                    )
            except Exception as e:
                properties.metadata["conductance_computation_error"] = str(e)

        # Spectral clustering if requested
        if compute_clustering and self._graph is not None:
            try:
                if n >= n_clusters:
                    labels = spectral_clustering(
                        self._graph,
                        k=n_clusters,
                        normalized=self._normalized,
                    )
                    properties.clustering = labels
                    properties.metadata["n_clusters_requested"] = n_clusters
                    properties.metadata["n_clusters_found"] = int(len(np.unique(labels)))
            except Exception as e:
                properties.metadata["clustering_error"] = str(e)

        # Add metadata
        properties.metadata["n_nodes"] = n
        properties.metadata["n_eigenvalues"] = len(properties.eigenvalues) if properties.eigenvalues is not None else 0
        properties.metadata["normalized_laplacian"] = self._normalized

        return properties

    def _create_graph_from_laplacian(self) -> nx.Graph:
        """Create a simple graph from Laplacian for compatibility with spectral_graph functions."""
        if self._laplacian is None:
            raise ValueError("No Laplacian matrix available")

        n = self._laplacian.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Extract edges from Laplacian (off-diagonal elements are -weight)
        L_coo = self._laplacian.tocoo()
        for i, j, v in zip(L_coo.row, L_coo.col, L_coo.data, strict=True):
            if i < j and v != 0:
                G.add_edge(i, j, weight=-v)

        return G

    def get_embedding(self, dim: int = 2) -> np.ndarray:
        """
        Compute spectral embedding of the graph.

        Uses the eigenvectors corresponding to the smallest non-zero
        eigenvalues to embed nodes in a low-dimensional space.

        Parameters
        ----------
        dim : int, default 2
            Dimension of the embedding space

        Returns
        -------
        numpy.ndarray
            Embedding coordinates of shape (n_nodes, dim)
        """
        from spectral_graph import spectral_embedding

        if self._graph is None:
            raise ValueError("Graph required for embedding computation")

        return spectral_embedding(self._graph, dim=dim, normalized=self._normalized)

    def compare_with(self, other: SpectralAnalyzer) -> dict[str, Any]:
        """
        Compare spectral properties with another analyzer.

        Useful for tracking changes in system state over time or
        comparing different configurations.

        Parameters
        ----------
        other : SpectralAnalyzer
            Another analyzer to compare against

        Returns
        -------
        dict
            Comparison metrics including eigenvalue differences,
            connectivity changes, etc.
        """
        props_self = self.analyze(k=min(10, self._n_nodes or 10))
        props_other = other.analyze(k=min(10, other._n_nodes or 10))

        comparison: dict[str, Any] = {
            "self_n_nodes": self._n_nodes,
            "other_n_nodes": other._n_nodes,
        }

        if props_self.eigenvalues is not None and props_other.eigenvalues is not None:
            min_len = min(len(props_self.eigenvalues), len(props_other.eigenvalues))
            eval_diff = props_self.eigenvalues[:min_len] - props_other.eigenvalues[:min_len]
            comparison["eigenvalue_rmse"] = float(np.sqrt(np.mean(eval_diff ** 2)))
            comparison["eigenvalue_max_diff"] = float(np.max(np.abs(eval_diff)))

        if props_self.algebraic_connectivity is not None and props_other.algebraic_connectivity is not None:
            comparison["algebraic_connectivity_diff"] = (
                props_self.algebraic_connectivity - props_other.algebraic_connectivity
            )

        if props_self.is_connected is not None and props_other.is_connected is not None:
            comparison["connectivity_changed"] = props_self.is_connected != props_other.is_connected

        return comparison


def analyze_quiesce_state(
    adjacency_matrix: sparse.spmatrix | np.ndarray | None = None,
    graph: nx.Graph | None = None,
    normalized: bool = False,
    k: int | None = None,
) -> SpectralProperties:
    """
    Convenience function to analyze quiesce system state spectrally.

    This function provides a simple interface for the main execution loop
    to invoke spectral analysis as a post-quiescence diagnostic step.

    Parameters
    ----------
    adjacency_matrix : scipy.sparse matrix or numpy.ndarray, optional
        Adjacency matrix representing the system state
    graph : networkx.Graph, optional
        Graph representing the system state
    normalized : bool, default False
        If True, use normalized Laplacian
    k : int, optional
        Number of eigenpairs to compute

    Returns
    -------
    SpectralProperties
        Computed spectral properties

    Raises
    ------
    ValueError
        If neither adjacency_matrix nor graph is provided

    Examples
    --------
    >>> import numpy as np
    >>> import networkx as nx
    >>> from quisce.spectral_analysis import analyze_quiesce_state
    >>>
    >>> # From graph
    >>> G = nx.barbell_graph(5, 2)
    >>> props = analyze_quiesce_state(graph=G)
    >>> print(f"Connected: {props.is_connected}")  # doctest: +SKIP

    >>> # From adjacency matrix
    >>> adj = nx.adjacency_matrix(nx.path_graph(10))
    >>> props = analyze_quiesce_state(adjacency_matrix=adj)
    >>> print(f"Algebraic connectivity: {props.algebraic_connectivity:.4f}")  # doctest: +SKIP
    """
    if graph is not None:
        analyzer = SpectralAnalyzer.from_graph(graph, normalized=normalized)
    elif adjacency_matrix is not None:
        analyzer = SpectralAnalyzer.from_adjacency_matrix(adjacency_matrix, normalized=normalized)
    else:
        raise ValueError("Either adjacency_matrix or graph must be provided")

    return analyzer.analyze(k=k)


if __name__ == "__main__":
    # Run validation tests
    print("Testing quisce.spectral_analysis module...")
    print("=" * 50)

    if not SPECTRAL_GRAPH_AVAILABLE:
        print("ERROR: spectral_graph package not available")
        exit(1)

    # Test 1: Basic graph analysis
    print("\nTest 1: Path graph analysis")
    G_path = nx.path_graph(10)
    analyzer = SpectralAnalyzer.from_graph(G_path)
    props = analyzer.analyze()
    print(props.summary())
    assert props.algebraic_connectivity is not None
    assert props.is_connected is True
    print("✓ Path graph analysis passed")

    # Test 2: Disconnected graph
    print("\nTest 2: Disconnected graph analysis")
    G_disconnected = nx.disjoint_union(nx.path_graph(5), nx.path_graph(5))
    analyzer2 = SpectralAnalyzer.from_graph(G_disconnected)
    props2 = analyzer2.analyze()
    print(props2.summary())
    assert props2.num_components is not None and props2.num_components >= 2
    assert props2.is_connected is False
    print("✓ Disconnected graph analysis passed")

    # Test 3: From adjacency matrix
    print("\nTest 3: Analysis from adjacency matrix")
    G_complete = nx.complete_graph(8)
    adj = nx.adjacency_matrix(G_complete)
    analyzer3 = SpectralAnalyzer.from_adjacency_matrix(adj)
    props3 = analyzer3.analyze()
    print(props3.summary())
    assert props3.algebraic_connectivity is not None
    # Complete graph K_n has algebraic connectivity = n
    assert abs(props3.algebraic_connectivity - 8.0) < 0.1
    print("✓ Adjacency matrix analysis passed")

    # Test 4: Spectral clustering
    print("\nTest 4: Spectral clustering")
    G_barbell = nx.barbell_graph(6, 0)
    analyzer4 = SpectralAnalyzer.from_graph(G_barbell)
    props4 = analyzer4.analyze(compute_clustering=True, n_clusters=2)
    print(props4.summary())
    assert props4.clustering is not None
    unique_labels = len(np.unique(props4.clustering))
    assert unique_labels == 2
    print("✓ Spectral clustering passed")

    # Test 5: Convenience function
    print("\nTest 5: Convenience function")
    props5 = analyze_quiesce_state(graph=nx.cycle_graph(7))
    print(props5.summary())
    assert props5.eigenvalues is not None
    print("✓ Convenience function passed")

    # Test 6: Normalized Laplacian
    print("\nTest 6: Normalized Laplacian")
    analyzer6 = SpectralAnalyzer.from_graph(G_path, normalized=True)
    props6 = analyzer6.analyze()
    print(props6.summary())
    # Normalized Laplacian eigenvalues should be in [0, 2]
    assert props6.eigenvalues is not None
    assert props6.eigenvalues.min() >= -1e-10
    assert props6.eigenvalues.max() <= 2.0 + 1e-6
    print("✓ Normalized Laplacian passed")

    print("\n" + "=" * 50)
    print("All quisce.spectral_analysis tests passed!")
