#!/usr/bin/env python3
"""
Demonstration of spectral graph analysis for dolphin model upgrades.

This script demonstrates the complete workflow:
1. Construct an adjacency matrix from the dolphin model's layer interactions
2. Compute the top-k eigenvalues and eigenvectors of the graph Laplacian
3. Output a structured report mapping spectral gaps to proposed logical upgrades

The analysis is performed without modifying the original model files.
"""

import json
import numpy as np

from spectral_graph import (
    DolphinModel,
    SpectralAnalyzer,
    adjacency_matrix,
    laplacian_matrix,
    analyze_and_report,
)


def create_sample_dolphin_model() -> DolphinModel:
    """Create a sample dolphin-like mesh model for demonstration."""
    # Create vertices representing a simplified dolphin shape
    # Body: elongated ellipsoid
    n_body = 30
    t = np.linspace(0, 2 * np.pi, n_body)
    
    # Body vertices (elliptical cross-section)
    body_x = 5 * np.cos(t)
    body_y = 2 * np.sin(t)
    body_z = np.zeros(n_body)
    
    # Tail vertices
    tail_vertices = [
        [6.0, 0.0, 0.0],
        [7.0, 1.0, 0.0],
        [7.0, -1.0, 0.0],
        [8.0, 0.5, 0.0],
        [8.0, -0.5, 0.0],
    ]
    
    # Dorsal fin
    fin_vertices = [
        [2.0, 0.0, 1.5],
        [3.0, 0.0, 2.0],
        [4.0, 0.0, 1.5],
    ]
    
    # Combine all vertices
    vertices = np.column_stack([body_x, body_y, body_z])
    tail_arr = np.array(tail_vertices)
    fin_arr = np.array(fin_vertices)
    vertices = np.vstack([vertices, tail_arr, fin_arr])
    
    # Create faces (triangulation)
    faces = []
    
    # Body faces (connect consecutive vertices)
    for i in range(n_body - 2):
        faces.append([i, i + 1, i + 2])
    
    # Connect body to tail
    for i in range(len(tail_vertices) - 2):
        idx = n_body + i
        faces.append([n_body - 1, idx, idx + 1])
    
    # Connect body to fin
    for i in range(len(fin_vertices) - 2):
        idx = n_body + len(tail_vertices) + i
        faces.append([10, 15, idx])  # Simplified connection
    
    faces = np.array(faces, dtype=np.int64)
    
    return DolphinModel(vertices, faces)


def demonstrate_adjacency_matrix(model: DolphinModel) -> None:
    """Step 1: Demonstrate adjacency matrix construction."""
    print("=" * 70)
    print("STEP 1: Construct Adjacency Matrix from Model Connectivity")
    print("=" * 70)
    
    # Convert model to graph
    G = model.to_graph()
    
    # Construct adjacency matrix
    A = adjacency_matrix(G)
    
    print(f"\nModel statistics:")
    print(f"  Vertices: {model.n_vertices}")
    print(f"  Edges: {len(model.edges)}")
    print(f"  Faces: {len(model.faces) if model.faces is not None else 0}")
    
    print(f"\nAdjacency matrix properties:")
    print(f"  Shape: {A.shape}")
    print(f"  Non-zero entries: {A.nnz}")
    print(f"  Matrix type: {type(A).__name__}")
    
    # Show sparsity pattern (first 10x10 block)
    print(f"\nAdjacency matrix (first 10x10 block):")
    dense_block = A[:10, :10].toarray()
    print(dense_block.astype(int))
    
    print("\n✓ Adjacency matrix successfully constructed from model connectivity")


def demonstrate_eigenvalue_computation(model: DolphinModel, k: int = 10) -> None:
    """Step 2: Compute top-k eigenvalues and eigenvectors."""
    print("\n" + "=" * 70)
    print(f"STEP 2: Compute Top-{k} Eigenvalues and Eigenvectors of Laplacian")
    print("=" * 70)
    
    # Create spectral analyzer
    analyzer = SpectralAnalyzer.from_dolphin_model(model)
    
    # Get sorted eigenvalues
    eigenvalues = analyzer.get_sorted_eigenvalues(k=k)
    
    # Get corresponding eigenvectors
    eigenvectors = analyzer.get_eigenvectors(k=k)
    
    print(f"\nLaplacian eigenvalues (λ₁ to λ_{k}):")
    print("-" * 50)
    for i, val in enumerate(eigenvalues, 1):
        marker = "← λ₁ (zero mode)" if i == 1 else \
                 "← λ₂ (algebraic connectivity)" if i == 2 else ""
        print(f"  λ_{i:2d} = {val:12.6f}  {marker}")
    
    print(f"\nEigenvector matrix shape: {eigenvectors.shape}")
    print(f"  Rows: {eigenvectors.shape[0]} (one per vertex)")
    print(f"  Columns: {eigenvectors.shape[1]} (one per eigenvalue)")
    
    # Compute spectral gap analysis
    gaps = np.diff(eigenvalues)
    max_gap_idx = np.argmax(gaps)
    
    print(f"\nSpectral gap analysis:")
    print(f"  Largest gap: Δλ_{max_gap_idx + 1} = {gaps[max_gap_idx]:.6f}")
    print(f"  Gap location: between λ_{max_gap_idx + 1} and λ_{max_gap_idx + 2}")
    
    # Algebraic connectivity
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    print(f"\nAlgebraic connectivity (λ₂): {lambda_2:.6f}")
    if lambda_2 < 0.1:
        print("  → Indicates potential bottlenecks in connectivity")
    elif lambda_2 < 0.5:
        print("  → Moderate connectivity")
    else:
        print("  → Good overall connectivity")
    
    print("\n✓ Eigenvalue computation completed successfully")


def demonstrate_upgrade_report(model: DolphinModel) -> dict:
    """Step 3: Generate structured upgrade report."""
    print("\n" + "=" * 70)
    print("STEP 3: Generate Structured Upgrade Report")
    print("=" * 70)
    
    # Generate the report
    report = analyze_and_report(model, output_format="print")
    
    # Also return as JSON for programmatic use
    report_json = json.loads(report.to_json())
    
    print("\n" + "=" * 70)
    print("JSON REPORT OUTPUT (for integration with other tools)")
    print("=" * 70)
    print(json.dumps(report_json, indent=2))
    
    return report_json


def main():
    """Run the complete demonstration."""
    print("\n" + "#" * 70)
    print("# SPECTRAL GRAPH ANALYSIS FOR DOLPHIN MODEL UPGRADES")
    print("#" * 70)
    print("\nThis demonstration shows how spectral analysis can identify")
    print("structural bottlenecks and suggest architectural improvements.\n")
    
    # Create sample model
    print("Creating sample DolphinModel...")
    model = create_sample_dolphin_model()
    print(f"Created model with {model.n_vertices} vertices\n")
    
    # Step 1: Adjacency matrix
    demonstrate_adjacency_matrix(model)
    
    # Step 2: Eigenvalue computation
    demonstrate_eigenvalue_computation(model, k=10)
    
    # Step 3: Upgrade report
    report = demonstrate_upgrade_report(model)
    
    # Summary
    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)
    print(f"\nAnalysis complete. Key findings:")
    print(f"  • Model has {report['model_info']['n_vertices']} vertices and {report['model_info']['n_edges']} edges")
    print(f"  • Algebraic connectivity: {report['algebraic_connectivity']:.6f}")
    print(f"  • Significant spectral gaps found: {len(report['significant_gaps'])}")
    print(f"  • Upgrade recommendations generated: {len(report['recommendations'])}")
    
    print("\nRecommendations by priority:")
    high_priority = [r for r in report['recommendations'] if r['priority'] == 'high']
    medium_priority = [r for r in report['recommendations'] if r['priority'] == 'medium']
    low_priority = [r for r in report['recommendations'] if r['priority'] == 'low']
    
    if high_priority:
        print(f"  HIGH:   {len(high_priority)} recommendation(s)")
    if medium_priority:
        print(f"  MEDIUM: {len(medium_priority)} recommendation(s)")
    if low_priority:
        print(f"  LOW:    {len(low_priority)} recommendation(s)")
    
    print("\n✓ All steps completed successfully!")
    print("\nThe spectral analysis module is ready for use on actual dolphin models.")
    print("To analyze your own model, replace create_sample_dolphin_model() with")
    print("your model loading code and run analyze_and_report().\n")


if __name__ == "__main__":
    main()
