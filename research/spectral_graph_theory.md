# Spectral Graph Theory

## Introduction

Spectral graph theory studies the properties of graphs through the eigenvalues and eigenvectors of matrices associated with the graph. This approach provides powerful tools for graph partitioning, clustering, and network analysis.

## Key Matrices

### Adjacency Matrix $A$

For a graph $G = (V, E)$ with $n$ vertices:

$$A_{ij} = \begin{cases} 1 & \text{if } (i,j) \in E \\ 0 & \text{otherwise} \end{cases}$$

Properties:
- Symmetric for undirected graphs
- Eigenvalues are real for undirected graphs
- Largest eigenvalue relates to graph connectivity

### Degree Matrix $D$

Diagonal matrix where:

$$D_{ii} = \text{degree of vertex } i = \sum_j A_{ij}$$

### Laplacian Matrix $L$

$$L = D - A$$

Properties:
- Positive semi-definite
- Smallest eigenvalue is always 0
- Number of zero eigenvalues equals number of connected components
- Second smallest eigenvalue (algebraic connectivity) measures how well-connected the graph is

### Normalized Laplacian $\mathcal{L}$

$$\mathcal{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

Properties:
- Eigenvalues in range $[0, 2]$
- Better for graphs with varying degree distributions

## Spectral Clustering Algorithm

1. Construct affinity/similarity matrix $A$
2. Compute Laplacian $L = D - A$
3. Compute first $k$ eigenvectors of $L$
4. Form matrix $U \in \mathbb{R}^{n \times k}$ with eigenvectors as columns
5. Normalize rows of $U$
6. Apply k-means clustering to rows of $U$

Complexity: $O(n^3)$ for dense eigendecomposition

## Applications

- **Community Detection**: Identify clusters in social networks
- **Graph Partitioning**: Divide graphs into balanced components
- **Image Segmentation**: Treat pixels as graph nodes
- **Dimensionality Reduction**: Laplacian eigenmaps

## Numerical Considerations

- For large sparse graphs, use `scipy.sparse.linalg.eigsh`
- Normalized Laplacian often more stable numerically
- Watch for numerical precision with near-zero eigenvalues

## References

1. Chung, F. R. K. (1997). *Spectral Graph Theory*. CBMS Regional Conference Series in Mathematics, Vol. 92.
2. Von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17(4), 395-416.
3. Spielman, D. A. (2007). Spectral graph theory and its applications. *FOCS '07*.
