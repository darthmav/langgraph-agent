#!/usr/bin/env python
"""
Demonstration script for spectral graph analysis on a DolphinModel.

This script instantiates a DolphinModel, passes it to SpectralAnalyzer,
and prints the first 10 sorted eigenvalues to stdout.
"""

import numpy as np

from spectral_graph import DolphinModel, SpectralAnalyzer


def main():
    # Create a simple mesh representing a dolphin-like structure
    # Using a path graph structure for simplicity (vertices in a line)
    n_vertices = 20
    vertices = np.array([[i, 0, 0] for i in range(n_vertices)], dtype=float)
    
    # Create faces connecting consecutive vertices (triangular strips)
    faces = []
    for i in range(n_vertices - 2):
        faces.append([i, i + 1, i + 2])
    faces = np.array(faces)
    
    # Instantiate the DolphinModel
    model = DolphinModel(vertices, faces)
    
    print(f"Created DolphinModel with {model.n_vertices} vertices")
    
    # Create SpectralAnalyzer from the model
    analyzer = SpectralAnalyzer.from_dolphin_model(model)
    
    # Get the first 10 sorted eigenvalues
    eigenvalues = analyzer.get_sorted_eigenvalues(k=10)
    
    print("\nFirst 10 sorted eigenvalues of the Laplacian:")
    for i, val in enumerate(eigenvalues):
        print(f"  λ_{i+1} = {val:.6f}")


if __name__ == "__main__":
    main()
