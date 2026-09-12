"""Tests for the DolphinModel and SpectralAnalyzer classes."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# The package is not installed; import it from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectral_graph.dolphin_model import DolphinModel, SpectralAnalyzer  # noqa: E402

TOL = 1e-9


class TestDolphinModel:
    """Tests for the DolphinModel class."""

    def test_create_from_vertices_only(self) -> None:
        """A DolphinModel can be created with just vertices."""
        vertices = np.array([[i, 0, 0] for i in range(5)], dtype=np.float64)
        model = DolphinModel(vertices)
        assert model.n_vertices == 5
        assert model.dimension == 3
        assert len(model.edges) == 0

    def test_create_with_faces(self) -> None:
        """A DolphinModel extracts edges from faces."""
        vertices = np.array([[i, 0, 0] for i in range(5)], dtype=np.float64)
        faces = np.array([[i, i + 1, i + 2] for i in range(3)], dtype=np.int64)
        model = DolphinModel(vertices, faces)
        assert model.n_vertices == 5
        assert len(model.faces) == 3
        assert len(model.edges) > 0

    def test_create_with_explicit_edges(self) -> None:
        """A DolphinModel accepts explicit edge connectivity."""
        vertices = np.array([[i, 0, 0] for i in range(4)], dtype=np.float64)
        edges = [(0, 1), (1, 2), (2, 3)]
        model = DolphinModel(vertices, edges=edges)
        assert model.edges == edges

    def test_invalid_vertices_ndim(self) -> None:
        """A 1D vertices array raises ValueError."""
        vertices = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D array"):
            DolphinModel(vertices)

    def test_invalid_face_indices(self) -> None:
        """Face indices out of bounds raise ValueError."""
        vertices = np.array([[i, 0, 0] for i in range(3)], dtype=np.float64)
        faces = np.array([[0, 1, 5]], dtype=np.int64)  # 5 is out of bounds
        with pytest.raises(ValueError, match="Face indices"):
            DolphinModel(vertices, faces)

    def test_to_graph(self) -> None:
        """to_graph() creates a NetworkX graph with correct nodes and edges."""
        vertices = np.array([[i, 0, 0] for i in range(4)], dtype=np.float64)
        edges = [(0, 1), (1, 2), (2, 3)]
        model = DolphinModel(vertices, edges=edges)
        G = model.to_graph()
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3

    def test_get_vertex_positions(self) -> None:
        """get_vertex_positions() returns a copy of vertices."""
        vertices = np.array([[i, 0, 0] for i in range(3)], dtype=np.float64)
        model = DolphinModel(vertices)
        positions = model.get_vertex_positions()
        assert np.array_equal(positions, vertices)
        # Verify it's a copy
        positions[0, 0] = 999
        assert model.vertices[0, 0] != 999

    def test_get_connectivity(self) -> None:
        """get_connectivity() returns a copy of edges."""
        vertices = np.array([[i, 0, 0] for i in range(3)], dtype=np.float64)
        edges = [(0, 1), (1, 2)]
        model = DolphinModel(vertices, edges=edges)
        connectivity = model.get_connectivity()
        assert connectivity == edges
        # Verify it's a copy
        connectivity.append((99, 99))
        assert (99, 99) not in model.edges


class TestSpectralAnalyzer:
    """Tests for the SpectralAnalyzer class."""

    def test_create_from_graph(self) -> None:
        """SpectralAnalyzer can be created from a NetworkX graph."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        assert analyzer.graph is G

    def test_reject_directed_graph(self) -> None:
        """A directed graph raises ValueError."""
        import networkx as nx

        G = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(ValueError, match="undirected"):
            SpectralAnalyzer(G)

    def test_from_dolphin_model(self) -> None:
        """from_dolphin_model() creates an analyzer from a mesh."""
        vertices = np.array([[i, 0, 0] for i in range(5)], dtype=np.float64)
        model = DolphinModel(vertices, edges=[(i, i + 1) for i in range(4)])
        analyzer = SpectralAnalyzer.from_dolphin_model(model)
        assert isinstance(analyzer, SpectralAnalyzer)
        assert analyzer.graph.number_of_nodes() == 5

    def test_sorted_eigenvalues_shape(self) -> None:
        """get_sorted_eigenvalues() returns correct shape."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        eigenvalues = analyzer.get_sorted_eigenvalues()
        assert eigenvalues.shape == (5,)

    def test_sorted_eigenvalues_ascending(self) -> None:
        """Eigenvalues are returned in ascending order."""
        import networkx as nx

        G = nx.path_graph(10)
        analyzer = SpectralAnalyzer(G)
        eigenvalues = analyzer.get_sorted_eigenvalues()
        assert np.all(np.diff(eigenvalues) >= -TOL)

    def test_first_eigenvalue_is_zero(self) -> None:
        """The first eigenvalue of a connected graph is ~0."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        eigenvalues = analyzer.get_sorted_eigenvalues()
        assert abs(eigenvalues[0]) < TOL

    def test_get_k_eigenvalues(self) -> None:
        """Can request a subset of eigenvalues."""
        import networkx as nx

        G = nx.path_graph(10)
        analyzer = SpectralAnalyzer(G)
        eigenvalues = analyzer.get_sorted_eigenvalues(k=3)
        assert eigenvalues.shape == (3,)

    def test_eigenvalues_k_too_large(self) -> None:
        """Requesting too many eigenvalues raises ValueError."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        with pytest.raises(ValueError, match="Cannot request"):
            analyzer.get_sorted_eigenvalues(k=10)

    def test_eigenvectors_shape(self) -> None:
        """get_eigenvectors() returns correct shape."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        evecs = analyzer.get_eigenvectors(k=3)
        assert evecs.shape == (5, 3)

    def test_spectral_coordinates_shape(self) -> None:
        """get_spectral_coordinates() returns correct shape."""
        import networkx as nx

        G = nx.path_graph(10)
        analyzer = SpectralAnalyzer(G)
        coords = analyzer.get_spectral_coordinates(dim=2)
        assert coords.shape == (10, 2)

    def test_spectral_coordinates_skip_constant_mode(self) -> None:
        """Spectral coordinates exclude the constant (first) eigenvector."""
        import networkx as nx

        G = nx.path_graph(10)
        analyzer = SpectralAnalyzer(G)
        coords = analyzer.get_spectral_coordinates(dim=2)
        # Get full eigenvectors to compare
        evecs = analyzer.get_eigenvectors(k=3)
        # coords should equal evecs[:, 1:3] (skipping first column)
        assert np.allclose(coords, evecs[:, 1:3])

    def test_energy_distribution_normalized(self) -> None:
        """Energy distribution sums to 1 when normalized."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        energy = analyzer.get_energy_distribution(normalize=True)
        assert abs(energy.sum() - 1.0) < TOL

    def test_energy_distribution_shape(self) -> None:
        """Energy distribution has correct shape."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        energy = analyzer.get_energy_distribution()
        assert energy.shape == (5,)

    def test_map_to_displacement_shape(self) -> None:
        """map_to_displacement() returns correct shape."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        displacement = analyzer.map_to_displacement(mode=1, amplitude=0.1)
        assert displacement.shape == (5, 3)

    def test_map_to_displacement_out_of_range(self) -> None:
        """Invalid mode index raises ValueError."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        with pytest.raises(ValueError, match="out of range"):
            analyzer.map_to_displacement(mode=10)

    def test_map_to_vertex_colors_shape(self) -> None:
        """map_to_vertex_colors() returns RGB colors."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        try:
            colors = analyzer.map_to_vertex_colors(mode=1)
            assert colors.shape == (5, 3)
            assert colors.dtype == np.uint8
            assert colors.min() >= 0
            assert colors.max() <= 255
        except ImportError:
            # matplotlib not available, fallback grayscale
            colors = analyzer.map_to_vertex_colors(mode=1)
            assert colors.shape == (5, 3)

    def test_clear_cache(self) -> None:
        """clear_cache() resets cached values."""
        import networkx as nx

        G = nx.path_graph(5)
        analyzer = SpectralAnalyzer(G)
        # Compute eigenvalues
        _ = analyzer.get_sorted_eigenvalues()
        assert analyzer._eigenvalues is not None
        # Clear cache
        analyzer.clear_cache()
        assert analyzer._eigenvalues is None

    def test_path_graph_closed_form_eigenvalues(self) -> None:
        """Eigenvalues of path graph match closed form: λ_k = 2 - 2*cos(π*k/n)."""
        import networkx as nx

        n = 10
        G = nx.path_graph(n)
        analyzer = SpectralAnalyzer(G)
        eigenvalues = analyzer.get_sorted_eigenvalues()
        expected = np.array([2 - 2 * np.cos(np.pi * k / n) for k in range(n)])
        assert np.allclose(eigenvalues, expected, atol=TOL)

    def test_cycle_graph_closed_form_eigenvalues(self) -> None:
        """Eigenvalues of cycle graph match closed form: λ_k = 2 - 2*cos(2π*k/n)."""
        import networkx as nx

        n = 8
        G = nx.cycle_graph(n)
        analyzer = SpectralAnalyzer(G)
        eigenvalues = analyzer.get_sorted_eigenvalues()
        expected = np.sort([2 - 2 * np.cos(2 * np.pi * k / n) for k in range(n)])
        assert np.allclose(eigenvalues, expected, atol=TOL)

    def test_complete_graph_eigenvalues(self) -> None:
        """Complete graph K_n has spectrum {0} ∪ {n} with multiplicity n-1."""
        import networkx as nx

        n = 6
        G = nx.complete_graph(n)
        analyzer = SpectralAnalyzer(G)
        eigenvalues = analyzer.get_sorted_eigenvalues()
        assert abs(eigenvalues[0]) < TOL
        assert np.allclose(eigenvalues[1:], float(n), atol=TOL)


if __name__ == "__main__":
    # Run simple validation
    print("Testing dolphin_model tests...")

    # Test DolphinModel
    vertices = np.array([[i, 0, 0] for i in range(5)], dtype=np.float64)
    model = DolphinModel(vertices)
    assert model.n_vertices == 5
    print("✓ DolphinModel basic creation works")

    # Test SpectralAnalyzer
    import networkx as nx

    G = nx.path_graph(5)
    analyzer = SpectralAnalyzer(G)
    eigenvalues = analyzer.get_sorted_eigenvalues()
    assert eigenvalues.shape == (5,)
    assert abs(eigenvalues[0]) < TOL
    print("✓ SpectralAnalyzer basic functionality works")

    print("\nAll dolphin_model tests passed!")
