"""
Spectral Clustering Example

Demonstrates graph partitioning using Laplacian eigenvectors.
Based on spectral graph theory principles.

References:
- Chung, F. R. K. (1997). Spectral Graph Theory.
- Von Luxburg, U. (2007). A tutorial on spectral clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def create_block_diagonal_graph(n_per_block=20, n_blocks=3, noise=0.1):
    """
    Create a graph with clear community structure.
    
    Parameters:
    -----------
    n_per_block : int
        Number of nodes per community
    n_blocks : int
        Number of communities
    noise : float
        Probability of edges between communities
    
    Returns:
    --------
    A : ndarray
        Adjacency matrix
    """
    n = n_per_block * n_blocks
    A = np.zeros((n, n))
    
    # Create dense blocks (communities)
    for i in range(n_blocks):
        start = i * n_per_block
        end = (i + 1) * n_per_block
        # Dense connections within community
        block = np.random.rand(n_per_block, n_per_block)
        block = (block > 0.3).astype(float)  # 70% density within community
        np.fill_diagonal(block, 0)  # No self-loops
        A[start:end, start:end] = block
    
    # Add sparse noise between communities
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            start_i, end_i = i * n_per_block, (i + 1) * n_per_block
            start_j, end_j = j * n_per_block, (j + 1) * n_per_block
            noise_matrix = (np.random.rand(n_per_block, n_per_block) < noise).astype(float)
            A[start_i:end_i, start_j:end_j] = noise_matrix
            A[start_j:end_j, start_i:end_i] = noise_matrix.T
    
    # Make symmetric
    A = (A + A.T) / 2
    
    return A


def compute_laplacian(A):
    """
    Compute the unnormalized Laplacian matrix.
    
    L = D - A
    
    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    
    Returns:
    --------
    L : ndarray
        Laplacian matrix
    D : ndarray
        Degree matrix (diagonal)
    """
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L, D


def compute_normalized_laplacian(A):
    """
    Compute the normalized Laplacian matrix.
    
    L_norm = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2)
    
    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    
    Returns:
    --------
    L_norm : ndarray
        Normalized Laplacian matrix
    """
    degrees = A.sum(axis=1)
    # Avoid division by zero
    degrees[degrees == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    return L_norm


def spectral_clustering(A, k=3):
    """
    Perform spectral clustering on adjacency matrix.
    
    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    k : int
        Number of clusters
    
    Returns:
    --------
    labels : ndarray
        Cluster assignments for each node
    eigenvectors : ndarray
        First k eigenvectors used for clustering
    eigenvalues : ndarray
        Corresponding eigenvalues
    """
    n = len(A)
    
    # Compute normalized Laplacian
    L_norm = compute_normalized_laplacian(A)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(L_norm)
    
    # Take first k eigenvectors (smallest eigenvalues)
    U = eigenvectors[:, :k]
    
    # Normalize rows
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1  # Avoid division by zero
    U_normalized = U / row_norms
    
    # Simple k-means on the eigenvector matrix
    labels = simple_kmeans(U_normalized, k)
    
    return labels, eigenvectors[:, :k], eigenvalues[:k]


def simple_kmeans(X, k, max_iters=100):
    """
    Simple k-means implementation for clustering.
    
    Parameters:
    -----------
    X : ndarray
        Data matrix (n_samples x n_features)
    k : int
        Number of clusters
    max_iters : int
        Maximum iterations
    
    Returns:
    --------
    labels : ndarray
        Cluster assignments
    """
    n = len(X)
    
    # Initialize centroids randomly
    np.random.seed(42)
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices].copy()
    
    labels = np.zeros(n, dtype=int)
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.zeros((n, k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)
        new_labels = np.argmin(distances, axis=1)
        
        # Check convergence
        if np.all(labels == new_labels):
            break
        labels = new_labels
        
        # Update centroids
        for j in range(k):
            mask = labels == j
            if np.sum(mask) > 0:
                centroids[j] = X[mask].mean(axis=0)
    
    return labels


def visualize_clustering(A, labels, title="Spectral Clustering Result"):
    """
    Visualize the clustering result by reordering adjacency matrix.
    
    Parameters:
    -----------
    A : ndarray
        Adjacency matrix
    labels : ndarray
        Cluster assignments
    title : str
        Plot title
    """
    # Sort nodes by cluster assignment
    sort_idx = np.argsort(labels)
    A_sorted = A[sort_idx][:, sort_idx]
    labels_sorted = labels[sort_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original adjacency matrix
    im1 = axes[0].imshow(A, cmap='Greys', interpolation='none')
    axes[0].set_title('Original Adjacency Matrix')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')
    plt.colorbar(im1, ax=axes[0])
    
    # Sorted adjacency matrix (showing clusters)
    im2 = axes[1].imshow(A_sorted, cmap='Greys', interpolation='none')
    axes[1].set_title('Sorted by Cluster Assignment')
    axes[1].set_xlabel('Node')
    axes[1].set_ylabel('Node')
    plt.colorbar(im2, ax=axes[1])
    
    # Mark cluster boundaries
    for ax in axes:
        cluster_bounds = []
        for i in range(len(labels_sorted) - 1):
            if labels_sorted[i] != labels_sorted[i + 1]:
                cluster_bounds.append(i + 0.5)
        for bound in cluster_bounds:
            ax.axhline(bound, color='red', linewidth=2)
            ax.axvline(bound, color='red', linewidth=2)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_eigenvalues(eigenvalues, k=10):
    """
    Plot the smallest eigenvalues of the Laplacian.
    
    Parameters:
    -----------
    eigenvalues : ndarray
        Eigenvalues (sorted)
    k : int
        Number of eigenvalues to show
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(k), eigenvalues[:k], color='steelblue')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Smallest Eigenvalues of Normalized Laplacian')
    ax.grid(True, alpha=0.3)
    return fig


def main():
    """Main demonstration of spectral clustering."""
    print("=" * 60)
    print("Spectral Clustering Demonstration")
    print("=" * 60)
    
    # Create graph with community structure
    np.random.seed(42)
    n_per_block = 20
    n_blocks = 3
    
    print(f"\nCreating graph with {n_blocks} communities, {n_per_block} nodes each...")
    A = create_block_diagonal_graph(n_per_block=n_per_block, n_blocks=n_blocks, noise=0.05)
    
    # Compute Laplacian eigenvalues
    L_norm = compute_normalized_laplacian(A)
    eigenvalues, _ = eigh(L_norm)
    
    print(f"\nGraph statistics:")
    print(f"  Total nodes: {len(A)}")
    print(f"  Total edges: {int(A.sum() / 2)}")
    print(f"  Density: {A.sum() / (len(A) * (len(A) - 1)):.3f}")
    
    print(f"\nSmallest eigenvalues of normalized Laplacian:")
    for i in range(min(6, len(eigenvalues))):
        print(f"  λ_{i} = {eigenvalues[i]:.6f}")
    
    # Perform spectral clustering
    print(f"\nPerforming spectral clustering with k={n_blocks}...")
    labels, eigenvectors, cluster_eigenvalues = spectral_clustering(A, k=n_blocks)
    
    # Check clustering accuracy
    print(f"\nClustering results:")
    for i in range(n_blocks):
        count = np.sum(labels == i)
        print(f"  Cluster {i}: {count} nodes")
    
    # Visualize
    fig1 = visualize_clustering(A, labels, 
                                title=f"Spectral Clustering (k={n_blocks})")
    
    fig2 = plot_eigenvalues(eigenvalues, k=15)
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Spectral clustering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
