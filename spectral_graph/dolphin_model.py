"""
Dolphin model mesh representation and spectral analysis.

This module provides:
- DolphinModel: A simple mesh geometry class representing vertices and connectivity
- SpectralAnalyzer: A utility class that computes eigenvalues/eigenvectors of the
  Laplacian matrix derived from mesh connectivity, enabling frequency-domain
  visualization without modifying core mesh data structures.

The implementation uses only numpy and scipy for linear algebra operations,
consistent with existing project dependencies.
"""

from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from spectral_graph.laplacian import laplacian_matrix


class DolphinModel:
    """
    A simple mesh geometry class representing a 3D model (e.g., a dolphin).

    This class stores vertex positions and face/edge connectivity information.
    It is designed to be compatible with spectral analysis via the SpectralAnalyzer
    class through composition - the analyzer accepts the mesh geometry as input
    without altering the core mesh data structures.

    Attributes
    ----------
    vertices : np.ndarray
        Vertex positions array of shape (n_vertices, 3) for 3D coordinates
    faces : np.ndarray, optional
        Face connectivity array of shape (n_faces, 3) for triangular faces
    edges : list of tuple, optional
        Edge connectivity as list of (i, j) vertex index pairs

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_graph.dolphin_model import DolphinModel
    >>> # Create a simple tetrahedron-like mesh
    >>> vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.5, 0.866, 0.0],
    ...     [0.5, 0.289, 0.816]
    ... ])
    >>> faces = np.array([
    ...     [0, 1, 2],
    ...     [0, 1, 3],
    ...     [0, 2, 3],
    ...     [1, 2, 3]
    ... ])
    >>> model = DolphinModel(vertices, faces)
    >>> model.n_vertices
    4
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
        edges: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        """
        Initialize a DolphinModel with vertex positions and connectivity.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex positions of shape (n_vertices, 3) for 3D coordinates,
            or (n_vertices, 2) for 2D, or (n_vertices, d) for d-dimensional
        faces : np.ndarray, optional
            Face connectivity of shape (n_faces, 3) for triangular meshes,
            or (n_faces, n) for n-gonal faces
        edges : list of tuple, optional
            Edge connectivity as list of (i, j) vertex index pairs.
            If not provided and faces are given, edges are extracted from faces.

        Raises
        ------
        ValueError
            If vertices is not a 2D array or if face/edge indices are out of bounds
        """
        vertices = np.asarray(vertices, dtype=np.float64)

        if vertices.ndim != 2:
            raise ValueError(
                f"vertices must be a 2D array of shape (n_vertices, d), "
                f"got shape {vertices.shape}"
            )

        self.vertices = vertices
        self.n_vertices = vertices.shape[0]
        self.dimension = vertices.shape[1]

        # Declared Optional rather than inferred from the first branch: the
        # `else` below stores None, which is a real state for a model given
        # edges directly.
        self.faces: Optional[np.ndarray] = None

        # Store faces if provided
        if faces is not None:
            faces = np.asarray(faces, dtype=np.int64)
            if faces.ndim != 2:
                raise ValueError(
                    f"faces must be a 2D array of shape (n_faces, n), "
                    f"got shape {faces.shape}"
                )
            # Validate face indices
            if faces.max() >= self.n_vertices or faces.min() < 0:
                raise ValueError(
                    f"Face indices must be in range [0, {self.n_vertices - 1}]"
                )
            self.faces = faces

        # Store or compute edges
        if edges is not None:
            self.edges = edges
        elif faces is not None:
            # Extract edges from faces
            self.edges = self._extract_edges_from_faces(faces)
        else:
            self.edges = []

    def _extract_edges_from_faces(
        self, faces: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        Extract unique edges from face connectivity.

        Parameters
        ----------
        faces : np.ndarray
            Face connectivity array

        Returns
        -------
        list of tuple
            List of unique edges as (i, j) pairs with i < j
        """
        edge_set = set()
        for face in faces:
            n = len(face)
            for i in range(n):
                for j in range(i + 1, n):
                    v1, v2 = int(face[i]), int(face[j])
                    # Store edges with smaller index first for uniqueness
                    edge = (min(v1, v2), max(v1, v2))
                    edge_set.add(edge)
        return sorted(edge_set)

    def to_graph(self) -> nx.Graph:
        """
        Convert the mesh connectivity to a NetworkX graph.

        The graph has vertices as nodes and mesh edges as graph edges.
        This enables using the spectral_graph module's functions.

        Returns
        -------
        nx.Graph
            NetworkX graph representing mesh connectivity

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_graph.dolphin_model import DolphinModel
        >>> vertices = np.array([[0, 0], [1, 0], [0, 1]])
        >>> faces = np.array([[0, 1, 2]])
        >>> model = DolphinModel(vertices, faces)
        >>> G = model.to_graph()
        >>> G.number_of_nodes()
        3
        >>> G.number_of_edges()
        3
        """
        G = nx.Graph()

        # Add nodes with position attributes
        for i, pos in enumerate(self.vertices):
            G.add_node(i, position=pos)

        # Add edges from connectivity
        for edge in self.edges:
            G.add_edge(edge[0], edge[1])

        return G

    def get_vertex_positions(self) -> np.ndarray:
        """
        Get vertex positions as a numpy array.

        Returns
        -------
        np.ndarray
            Vertex positions of shape (n_vertices, d)
        """
        return self.vertices.copy()

    def get_connectivity(self) -> list[tuple[int, int]]:
        """
        Get edge connectivity as a list of vertex index pairs.

        Returns
        -------
        list of tuple
            List of (i, j) edge pairs
        """
        return list(self.edges)


class SpectralAnalyzer:
    """
    Spectral analysis utility for mesh geometry.

    This class computes the eigenvalues and eigenvectors of the Laplacian
    matrix derived from mesh connectivity, enabling frequency-domain
    visualization. It accepts mesh geometry (via DolphinModel or directly
    as a graph) without altering the core mesh data structures.

    The analyzer outputs:
    - Sorted eigenvalues (spectral energy distribution)
    - Eigenvectors (spectral coordinates for embedding)
    - Methods to map spectral properties to vertex colors or displacement

    Attributes
    ----------
    graph : nx.Graph
        The graph representation of the mesh connectivity
    eigenvalues : np.ndarray, optional
        Cached eigenvalues after computation
    eigenvectors : np.ndarray, optional
        Cached eigenvectors after computation

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_graph.dolphin_model import DolphinModel, SpectralAnalyzer
    >>> # Create a simple path-like mesh
    >>> vertices = np.array([[i, 0, 0] for i in range(5)])
    >>> faces = np.array([[i, i+1, i+2] for i in range(3)])
    >>> model = DolphinModel(vertices, faces)
    >>> analyzer = SpectralAnalyzer.from_dolphin_model(model)
    >>> eigenvalues = analyzer.get_sorted_eigenvalues()
    >>> eigenvalues.shape[0]
    5
    >>> # First eigenvalue should be ~0 (Laplacian property)
    >>> abs(eigenvalues[0]) < 1e-10
    True
    """

    def __init__(self, graph: nx.Graph) -> None:
        """
        Initialize the SpectralAnalyzer with a graph.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph representing mesh connectivity

        Raises
        ------
        ValueError
            If the graph is directed (spectral analysis requires undirected graphs)
        """
        if graph.is_directed():
            raise ValueError(
                "SpectralAnalyzer requires an undirected graph. "
                "Convert directed graphs with G.to_undirected()."
            )
        self.graph = graph
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None

    @classmethod
    def from_dolphin_model(cls, model: DolphinModel) -> SpectralAnalyzer:
        """
        Create a SpectralAnalyzer from a DolphinModel.

        This is the primary interface for analyzing dolphin mesh models.

        Parameters
        ----------
        model : DolphinModel
            The dolphin model mesh to analyze

        Returns
        -------
        SpectralAnalyzer
            Analyzer instance configured for the model's connectivity

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_graph.dolphin_model import DolphinModel, SpectralAnalyzer
        >>> vertices = np.array([[i, 0, 0] for i in range(4)])
        >>> model = DolphinModel(vertices)
        >>> analyzer = SpectralAnalyzer.from_dolphin_model(model)
        >>> isinstance(analyzer, SpectralAnalyzer)
        True
        """
        graph = model.to_graph()
        return cls(graph)

    def _compute_eigenpairs(self, k: Optional[int] = None) -> None:
        """
        Compute eigenvalues and eigenvectors of the Laplacian matrix.

        Uses sparse solver for large graphs, dense for small ones.
        Results are cached in self._eigenvalues and self._eigenvectors.

        Parameters
        ----------
        k : int, optional
            Number of eigenpairs to compute. If None, computes all.
        """
        n = self.graph.number_of_nodes()

        if k is None or k >= n:
            k = n

        # Build Laplacian matrix
        L = laplacian_matrix(self.graph)

        # Choose solver based on size
        if n < 50 or k >= n - 1:
            # Dense solver for small graphs or when computing most eigenvalues
            L_dense = L.toarray()
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(L_dense)
        else:
            # Sparse solver for large graphs with few eigenvalues
            # Use shift-invert for better convergence on smallest eigenvalues
            try:
                scale = float(np.abs(L.diagonal()).max()) or 1.0
                sigma = -1e-6 * scale
                evals, evecs = eigsh(L, k=k, sigma=sigma, which="LM")
            except (RuntimeError, MemoryError):
                # Fallback to standard smallest magnitude
                evals, evecs = eigsh(L, k=k, which="SM")

            # Sort by eigenvalue
            idx = np.argsort(evals)
            self._eigenvalues = evals[idx]
            self._eigenvectors = evecs[:, idx]

        # Ensure we have all eigenvalues if requested
        if k == n and self._eigenvalues.shape[0] < n:
            # Recompute with dense solver
            L_dense = L.toarray()
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(L_dense)

    def get_sorted_eigenvalues(self, k: Optional[int] = None) -> np.ndarray:
        """
        Return the sorted eigenvalues of the Laplacian matrix.

        This is the primary method for obtaining the spectral energy
        distribution, which can be used for frequency-domain visualization.

        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to return. If None, returns all.
            Eigenvalues are always returned in ascending order.

        Returns
        -------
        np.ndarray
            Sorted eigenvalues in ascending order (λ₁ ≤ λ₂ ≤ ... ≤ λₙ)
            The first eigenvalue λ₁ ≈ 0 corresponds to the constant mode.

        Raises
        ------
        ValueError
            If k is larger than the number of vertices

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from spectral_graph.dolphin_model import SpectralAnalyzer
        >>> G = nx.path_graph(5)
        >>> analyzer = SpectralAnalyzer(G)
        >>> eigenvalues = analyzer.get_sorted_eigenvalues()
        >>> eigenvalues.shape
        (5,)
        >>> # Verify ascending order
        >>> np.all(np.diff(eigenvalues) >= -1e-10)
        True
        >>> # First eigenvalue is ~0
        >>> abs(eigenvalues[0]) < 1e-10
        True
        """
        n = self.graph.number_of_nodes()

        if k is not None:
            if k > n:
                raise ValueError(
                    f"Cannot request {k} eigenvalues from graph with {n} nodes"
                )
            if k <= 0:
                return np.array([], dtype=np.float64)

        # Compute if not cached or if requesting more than cached
        if self._eigenvalues is None or (k is not None and k > len(self._eigenvalues)):
            self._compute_eigenpairs(k=k)

        # Read the cache into a local after the computation: the attribute is
        # Optional, and `_compute_eigenpairs` either fills it or raises. The
        # check is what says that here rather than leaving a reader -- or a
        # type checker -- to trust the call above.
        eigenvalues = self._eigenvalues
        if eigenvalues is None:  # pragma: no cover - the compute path raises first
            raise RuntimeError("eigenvalue computation left the cache empty")

        if k is None:
            return eigenvalues.copy()
        else:
            return eigenvalues[:k].copy()

    def get_eigenvectors(self, k: Optional[int] = None) -> np.ndarray:
        """
        Return the eigenvectors of the Laplacian matrix.

        The eigenvectors provide spectral coordinates that can be used
        for mesh embedding or deformation.

        Parameters
        ----------
        k : int, optional
            Number of eigenvectors to return. If None, returns all.
            Columns correspond to eigenvalues in ascending order.

        Returns
        -------
        np.ndarray
            Eigenvector matrix of shape (n_vertices, k) where column i
            is the eigenvector corresponding to the i-th eigenvalue.

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from spectral_graph.dolphin_model import SpectralAnalyzer
        >>> G = nx.path_graph(5)
        >>> analyzer = SpectralAnalyzer(G)
        >>> evecs = analyzer.get_eigenvectors(k=3)
        >>> evecs.shape
        (5, 3)
        """
        n = self.graph.number_of_nodes()

        if k is None:
            k = n

        if k > n:
            raise ValueError(
                f"Cannot request {k} eigenvectors from graph with {n} nodes"
            )

        # Compute if not cached or if requesting more than cached
        if self._eigenvectors is None or k > self._eigenvectors.shape[1]:
            self._compute_eigenpairs(k=k)

        # Same as `get_eigenvalues`: the cache is Optional and the computation
        # above fills it or raises.
        eigenvectors = self._eigenvectors
        if eigenvectors is None:  # pragma: no cover - the compute path raises first
            raise RuntimeError("eigenvector computation left the cache empty")

        return eigenvectors[:, :k].copy()

    def get_spectral_coordinates(self, dim: int = 2) -> np.ndarray:
        """
        Get spectral embedding coordinates for visualization.

        Uses the first non-trivial eigenvectors (excluding the constant
        mode) as coordinates for low-dimensional embedding. This is
        useful for visualizing the mesh structure in frequency domain.

        Parameters
        ----------
        dim : int, default 2
            Dimension of the embedding (typically 2 or 3)

        Returns
        -------
        np.ndarray
            Spectral coordinates of shape (n_vertices, dim)

        Raises
        ------
        ValueError
            If dim is too large for the graph size

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from spectral_graph.dolphin_model import SpectralAnalyzer
        >>> G = nx.path_graph(10)
        >>> analyzer = SpectralAnalyzer(G)
        >>> coords = analyzer.get_spectral_coordinates(dim=2)
        >>> coords.shape
        (10, 2)
        """
        n = self.graph.number_of_nodes()

        if dim >= n:
            raise ValueError(
                f"Cannot embed {n} nodes in {dim} dimensions using spectral coordinates "
                f"(need at least {dim + 1} nodes for {dim}-dimensional embedding)"
            )

        # Get eigenvectors, skip the first (constant) mode
        # Use eigenvectors 1 through dim (indices 1 to dim inclusive)
        evecs = self.get_eigenvectors(k=dim + 1)
        return evecs[:, 1 : dim + 1]

    def get_energy_distribution(self, normalize: bool = True) -> np.ndarray:
        """
        Compute the spectral energy distribution.

        The energy distribution shows how much "energy" (variance) is
        captured at each frequency (eigenvalue). This is useful for
        determining how many modes are needed for accurate reconstruction.

        Parameters
        ----------
        normalize : bool, default True
            If True, normalize energies to sum to 1

        Returns
        -------
        np.ndarray
            Energy values for each eigenvalue mode, shape (n_vertices,)

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from spectral_graph.dolphin_model import SpectralAnalyzer
        >>> G = nx.path_graph(5)
        >>> analyzer = SpectralAnalyzer(G)
        >>> energy = analyzer.get_energy_distribution()
        >>> energy.shape
        (5,)
        >>> # Normalized energy should sum to 1
        >>> abs(energy.sum() - 1.0) < 1e-10
        True
        """
        eigenvalues = self.get_sorted_eigenvalues()

        # Energy is proportional to eigenvalue magnitude
        # Skip the zero eigenvalue (constant mode has no energy in this sense)
        energy = eigenvalues.copy()
        energy[0] = 0.0  # Constant mode has zero energy

        if normalize and energy.sum() > 0:
            energy = energy / energy.sum()

        return energy

    def map_to_vertex_colors(
        self, mode: int = 1, colormap: str = "viridis"
    ) -> np.ndarray:
        """
        Map a spectral mode to vertex colors for visualization.

        This enables frequency-domain visualization by coloring vertices
        according to their value in a specific eigenvector mode.

        Parameters
        ----------
        mode : int, default 1
            Which eigenvector mode to use (0 = constant, 1 = Fiedler, etc.)
        colormap : str, default "viridis"
            Matplotlib colormap name for color mapping

        Returns
        -------
        np.ndarray
            RGB colors for each vertex, shape (n_vertices, 3)
            Values in range [0, 255]

        Raises
        ------
        ImportError
            If matplotlib is not available for colormap
        ValueError
            If mode index is out of range

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from spectral_graph.dolphin_model import SpectralAnalyzer
        >>> G = nx.path_graph(5)
        >>> analyzer = SpectralAnalyzer(G)
        >>> try:
        ...     colors = analyzer.map_to_vertex_colors(mode=1)
        ...     colors.shape
        ... except ImportError:
        ...     print("matplotlib not available")
        (5, 3)
        """
        n = self.graph.number_of_nodes()

        if mode < 0 or mode >= n:
            raise ValueError(
                f"Mode {mode} is out of range for graph with {n} nodes"
            )

        # Get the eigenvector for this mode
        evecs = self.get_eigenvectors(k=mode + 1)
        values = evecs[:, mode]

        # Normalize to [0, 1]
        vmin, vmax = values.min(), values.max()
        if vmax - vmin > 1e-10:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.ones(n) * 0.5

        # Apply colormap
        try:
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(colormap)
            # Through `asarray` because matplotlib ships no stubs, so the
            # colormap's return is Any and would carry that out of a function
            # declared to return an array.
            rgba = np.asarray(cmap(normalized))
            # Convert to RGB (drop alpha) and scale to 0-255
            rgb = (rgba[:, :3] * 255).astype(np.uint8)
            return rgb
        except ImportError:
            # Fallback: grayscale without matplotlib
            rgb = (normalized * 255).astype(np.uint8)
            return np.stack([rgb, rgb, rgb], axis=1)

    def map_to_displacement(
        self, mode: int = 1, amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Map a spectral mode to vertex displacements for deformation.

        This enables frequency-domain deformation by displacing vertices
        along their normal direction (or a specified axis) according to
        their value in a specific eigenvector mode.

        Parameters
        ----------
        mode : int, default 1
            Which eigenvector mode to use for displacement
        amplitude : float, default 1.0
            Scaling factor for displacement magnitude

        Returns
        -------
        np.ndarray
            Displacement vectors for each vertex, shape (n_vertices, 3)
            Can be added to original vertex positions for deformation

        Raises
        ------
        ValueError
            If mode index is out of range

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_graph.dolphin_model import DolphinModel, SpectralAnalyzer
        >>> vertices = np.array([[i, 0, 0] for i in range(5)], dtype=float)
        >>> model = DolphinModel(vertices)
        >>> analyzer = SpectralAnalyzer.from_dolphin_model(model)
        >>> displacement = analyzer.map_to_displacement(mode=1, amplitude=0.1)
        >>> displacement.shape
        (5, 3)
        """
        n = self.graph.number_of_nodes()

        if mode < 0 or mode >= n:
            raise ValueError(
                f"Mode {mode} is out of range for graph with {n} nodes"
            )

        # Get the eigenvector for this mode
        evecs = self.get_eigenvectors(k=mode + 1)
        values = evecs[:, mode] * amplitude

        # Return as 3D displacement (apply to y-axis by default)
        # For a full mesh, this could be applied along vertex normals
        displacement = np.zeros((n, 3), dtype=np.float64)
        displacement[:, 1] = values  # Displace along y-axis

        return displacement

    def clear_cache(self) -> None:
        """
        Clear cached eigenvalues and eigenvectors.

        Useful when the underlying graph has changed or to free memory.
        """
        self._eigenvalues = None
        self._eigenvectors = None


if __name__ == "__main__":
    # Run validation tests
    import networkx as nx

    print("Testing dolphin_model.py...")

    # Test DolphinModel creation
    vertices = np.array([[i, 0, 0] for i in range(5)], dtype=np.float64)
    faces = np.array([[i, i + 1, i + 2] for i in range(3)], dtype=np.int64)
    model = DolphinModel(vertices, faces)
    assert model.n_vertices == 5
    assert model.dimension == 3
    assert len(model.edges) > 0
    print("✓ DolphinModel creation works correctly")

    # Test graph conversion
    G = model.to_graph()
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == len(model.edges)
    print("✓ DolphinModel.to_graph() works correctly")

    # Test SpectralAnalyzer
    analyzer = SpectralAnalyzer.from_dolphin_model(model)
    eigenvalues = analyzer.get_sorted_eigenvalues()
    assert eigenvalues.shape == (5,)
    assert abs(eigenvalues[0]) < 1e-10  # First eigenvalue is ~0
    assert np.all(np.diff(eigenvalues) >= -1e-10)  # Ascending order
    print("✓ SpectralAnalyzer.get_sorted_eigenvalues() works correctly")

    # Test eigenvectors
    evecs = analyzer.get_eigenvectors(k=3)
    assert evecs.shape == (5, 3)
    print("✓ SpectralAnalyzer.get_eigenvectors() works correctly")

    # Test spectral coordinates
    coords = analyzer.get_spectral_coordinates(dim=2)
    assert coords.shape == (5, 2)
    print("✓ SpectralAnalyzer.get_spectral_coordinates() works correctly")

    # Test energy distribution
    energy = analyzer.get_energy_distribution()
    assert energy.shape == (5,)
    assert abs(energy.sum() - 1.0) < 1e-10  # Normalized
    print("✓ SpectralAnalyzer.get_energy_distribution() works correctly")

    # Test displacement mapping
    displacement = analyzer.map_to_displacement(mode=1, amplitude=0.1)
    assert displacement.shape == (5, 3)
    print("✓ SpectralAnalyzer.map_to_displacement() works correctly")

    # Test on a known graph (path graph)
    G_path = nx.path_graph(10)
    analyzer_path = SpectralAnalyzer(G_path)
    eigenvalues_path = analyzer_path.get_sorted_eigenvalues()

    # For path graph P_n: λ_k = 2 - 2*cos(π*k/n)
    expected = np.array([2 - 2 * np.cos(np.pi * k / 10) for k in range(10)])
    assert np.allclose(eigenvalues_path, expected, atol=1e-10)
    print("✓ SpectralAnalyzer produces correct eigenvalues for path graph")

    print("\nAll dolphin_model.py tests passed!")
