"""
Graph Spectrum Visualization

Plots eigenvalue distributions for different graph types to illustrate
how graph structure affects the spectral properties.

References:
- Chung, F. R. K. (1997). Spectral Graph Theory.
- Spielman, D. A. (2007). Spectral graph theory and its applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def create_complete_graph(n):
    """
    Create adjacency matrix for complete graph K_n.
    Every node connected to every other node.
    """
    A = np.ones((n, n)) - np.eye(n)
    return A


def create_cycle_graph(n):
    """
    Create adjacency matrix for cycle graph C_n.
    Nodes connected in a ring.
    """
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i - 1) % n] = 1
        A[i, (i + 1) % n] = 1
    return A


def create_path_graph(n):
    """
    Create adjacency matrix for path graph P_n.
    Nodes connected in a line.
    """
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


def create_star_graph(n):
    """
    Create adjacency matrix for star graph S_n.
    One central node connected to all others.
    """
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = 1
        A[i, 0] = 1
    return A


def create_random_graph(n, p=0.3):
    """
    Create Erdos-Renyi random graph G(n, p).
    """
    np.random.seed(42)
    A = (np.random.rand(n, n) < p).astype(float)
    A = np.triu(A, 1)
    A = A + A.T  # Make symmetric
    return A


def create_community_graph(n_per_block=15, n_blocks=3, intra_p=0.7, inter_p=0.05):
    """
    Create graph with community structure.
    """
    n = n_per_block * n_blocks
    A = np.zeros((n, n))
    
    # Dense connections within communities
    for i in range(n_blocks):
        start = i * n_per_block
        end = (i + 1) * n_per_block
        block = (np.random.rand(n_per_block, n_per_block) < intra_p).astype(float)
        block = np.triu(block, 1)
        block = block + block.T
        A[start:end, start:end] = block
    
    # Sparse connections between communities
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            start_i, end_i = i * n_per_block, (i + 1) * n_per_block
            start_j, end_j = j * n_per_block, (j + 1) * n_per_block
            noise = (np.random.rand(n_per_block, n_per_block) < inter_p).astype(float)
            A[start_i:end_i, start_j:end_j] = noise
            A[start_j:end_j, start_i:end_i] = noise.T
    
    return A


def compute_laplacian_spectrum(A):
    """
    Compute eigenvalues of the normalized Laplacian.
    
    Returns sorted eigenvalues.
    """
    n = len(A)
    degrees = A.sum(axis=1)
    
    # Handle isolated nodes
    degrees[degrees == 0] = 1
    
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    eigenvalues, _ = eigh(L_norm)
    return np.sort(eigenvalues)


def plot_spectrum(eigenvalues, graph_name, ax):
    """
    Plot eigenvalue spectrum.
    """
    n = len(eigenvalues)
    
    # Plot individual eigenvalues
    ax.scatter(range(n), eigenvalues, s=20, alpha=0.7, color='steelblue')
    ax.plot(range(n), eigenvalues, alpha=0.5, color='steelblue')
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'{graph_name}\nλ ∈ [{eigenvalues[0]:.3f}, {eigenvalues[-1]:.3f}]')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 2.1)  # Normalized Laplacian eigenvalues in [0, 2]


def plot_histogram(eigenvalues, graph_name, ax):
    """
    Plot histogram of eigenvalues.
    """
    ax.hist(eigenvalues, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Count')
    ax.set_title(f'{graph_name}\nDistribution')
    ax.grid(True, alpha=0.3)


def main():
    """Main demonstration of graph spectrum visualization."""
    print("=" * 60)
    print("Graph Spectrum Visualization")
    print("=" * 60)
    
    n = 30  # Number of nodes for most graphs
    
    # Create different graph types
    graphs = {
        'Complete Graph K_30': create_complete_graph(n),
        'Cycle Graph C_30': create_cycle_graph(n),
        'Path Graph P_30': create_path_graph(n),
        'Star Graph S_30': create_star_graph(n),
        'Random Graph G(30, 0.3)': create_random_graph(n, p=0.3),
        'Community Graph (3 blocks)': create_community_graph(
            n_per_block=10, n_blocks=3, intra_p=0.8, inter_p=0.03
        )
    }
    
    # Compute spectra
    spectra = {}
    for name, A in graphs.items():
        eigenvalues = compute_laplacian_spectrum(A)
        spectra[name] = eigenvalues
        print(f"\n{name}:")
        print(f"  Nodes: {n}, Edges: {int(A.sum() / 2)}")
        print(f"  Eigenvalue range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
        print(f"  Algebraic connectivity (λ_1): {eigenvalues[1]:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, eigenvalues) in enumerate(spectra.items()):
        plot_spectrum(eigenvalues, name, axes[idx])
    
    plt.tight_layout()
    plt.savefig('examples/graph_spectra_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved spectrum comparison plot to examples/graph_spectra_comparison.png")
    
    # Create histogram figure
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    for idx, (name, eigenvalues) in enumerate(spectra.items()):
        plot_histogram(eigenvalues, name, axes2[idx])
    
    plt.tight_layout()
    plt.savefig('examples/graph_spectra_histograms.png', dpi=150, bbox_inches='tight')
    print("Saved histogram plot to examples/graph_spectra_histograms.png")
    
    # Show plots
    plt.show()
    
    print("\n" + "=" * 60)
    print("Key Observations:")
    print("-" * 60)
    print("1. Complete graph: All non-zero eigenvalues are equal (= n/(n-1))")
    print("2. Cycle/Path graphs: Eigenvalues follow predictable patterns")
    print("3. Star graph: One large eigenvalue, rest clustered near 1")
    print("4. Random graph: More uniform distribution")
    print("5. Community graph: Small eigenvalues indicate community structure")
    print("=" * 60)


if __name__ == "__main__":
    main()
