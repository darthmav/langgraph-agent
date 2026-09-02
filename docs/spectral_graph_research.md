# Spectral Graph Theory Research Notes

## Overview

This document summarizes key concepts in spectral graph theory and the Python ecosystem for implementing them. Research conducted for the `spectral_graph` package implementation.

---

## Part 1: Spectral Graph Mathematics

### 1.1 Adjacency and Laplacian Matrices

**Adjacency Matrix (A):**
For a graph G = (V, E) with n vertices, the adjacency matrix A is an n×n matrix where:
- A[i,j] = 1 if (i,j) ∈ E, 0 otherwise (unweighted)
- A[i,j] = w_ij for weighted graphs

**Degree Matrix (D):**
Diagonal matrix where D[i,i] = degree of vertex i (sum of weights for weighted graphs).

**Unnormalized Laplacian (L):**
```
L = D - A
```
Properties:
- Positive semi-definite
- Eigenvalues: 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ
- λ₂ (algebraic connectivity/Fiedler value) > 0 iff graph is connected
- Multiplicity of eigenvalue 0 equals number of connected components

**Normalized Laplacian (L_norm or L_sym):**
```
L_norm = I - D^(-1/2) A D^(-1/2) = D^(-1/2) L D^(-1/2)
```
Properties:
- Eigenvalues in [0, 2]
- Better for graphs with heterogeneous degree distributions
- Used in spectral clustering and random walk analysis

**Random Walk Laplacian (L_rw):**
```
L_rw = I - D^(-1) A = D^(-1) L
```
- Shares eigenvalues with L_norm
- Used in PageRank and diffusion processes

### 1.2 Fiedler Vector and Algebraic Connectivity

**Fiedler Value (λ₂):**
The second smallest eigenvalue of L. Also called algebraic connectivity.
- Measures how well-connected the graph is
- Larger λ₂ → harder to disconnect the graph

**Fiedler Vector:**
The eigenvector corresponding to λ₂.
- Used for spectral bipartitioning
- Sign pattern of entries suggests a natural 2-way cut
- Nodes with positive entries vs. negative entries form two clusters

**Key Reference:**
- Fiedler, M. (1973). "Algebraic Connectivity of Graphs". Czechoslovak Mathematical Journal.
- https://dml.cz/handle/10338.dmlcz/101162

### 1.3 Spectral Clustering

Algorithm:
1. Compute k smallest eigenvectors of L (or L_norm)
2. Form matrix U ∈ R^(n×k) with eigenvectors as columns
3. Normalize rows of U to unit length (for L_norm)
4. Apply k-means to rows of U
5. Assign node i to cluster of row i

**Why it works:**
- For k disconnected components, the k smallest eigenvectors are indicator vectors
- For nearly disconnected graphs, eigenvectors are "smooth" within components

**Key Reference:**
- von Luxburg, U. (2007). "A Tutorial on Spectral Clustering". Statistics and Computing.
- https://link.springer.com/article/10.1007/s11222-007-9073-2
- Ng, Jordan, Weiss (2002). "On Spectral Clustering: Analysis and an Algorithm". NIPS.
- https://papers.nips.cc/paper/2001/hash/801272ee79cfde795eef78aed585cc15-Abstract.html

### 1.4 Cheeger Inequality

**Cheeger Constant (isoperimetric number):**
```
h(G) = min_{S: |S|≤n/2} |∂S| / |S|
```
where ∂S is the set of edges crossing the cut (S, V\S).

**Cheeger Inequality:**
```
λ₂/2 ≤ h(G) ≤ √(2λ₂)
```

This relates the spectral gap (λ₂) to the combinatorial expansion (h(G)).

**Key Reference:**
- Cheeger, J. (1970). "A lower bound for the smallest eigenvalue of the Laplacian".
- Alon, N., Milman, V. (1985). "λ₁, Isoperimetric inequalities for graphs".
- https://www.sciencedirect.com/science/article/pii/0095895685900929

### 1.5 Spectral Embedding

Map each node i to a point in R^k using the k smallest non-trivial eigenvectors:
```
embedding(i) = (u₂[i], u₃[i], ..., u_{k+1}[i])
```

This preserves graph structure: connected nodes tend to be close in embedding space.

---

## Part 2: Python Ecosystem

### 2.1 NumPy/SciPy for Eigenvalue Computation

**Dense eigensolvers (numpy.linalg, scipy.linalg):**
- `numpy.linalg.eigh(A)` - symmetric/Hermitian matrices
- `scipy.linalg.eigh(A, B)` - generalized eigenvalue problem
- Use for small to medium graphs (n < ~1000)

**Sparse eigensolvers (scipy.sparse.linalg):**
- `scipy.sparse.linalg.eigsh(A, k, which='SM')` - k smallest eigenvalues
- `scipy.sparse.linalg.eigs(A, k, which='SR')` - for non-symmetric
- Uses ARPACK (Lanczos/Arnoldi iteration)
- Essential for large sparse graphs (n > ~1000)

**Key Documentation:**
- SciPy sparse.linalg.eigsh: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
- ARPACK documentation: https://www.caam.rice.edu/software/ARPACK/

### 2.2 NetworkX for Graph Construction

NetworkX provides:
- Graph construction utilities (various random graph models)
- `nx.laplacian_matrix(G)` - unnormalized Laplacian
- `nx.normalized_laplacian_matrix(G)` - normalized Laplacian
- `nx.adjacency_matrix(G)` - adjacency matrix
- `nx.fiedler_vector(G)` - for validation only

**Important:** We use NetworkX only for graph construction and validation, not as a substitute for our implementations.

**Key Documentation:**
- NetworkX spectral module: https://networkx.org/documentation/stable/reference/linalg.html

### 2.3 Scikit-learn for Clustering

- `sklearn.cluster.KMeans` - used in spectral clustering pipeline
- `sklearn.cluster.SpectralClustering` - built-in implementation for comparison

**Key Documentation:**
- KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- SpectralClustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

---

## Part 3: Implementation Design

### Module Structure

```
spectral_graph/
├── __init__.py          # Package exports
├── laplacian.py         # Laplacian construction (L, L_norm, L_rw)
├── spectrum.py          # Eigenvalue/eigenvector computation
├── fiedler.py           # Fiedler vector and bipartitioning
├── embedding.py         # Spectral embedding
└── clustering.py        # Spectral clustering, conductance, sweep cut,
                         # Cheeger bounds
```

Cheeger lives in `clustering.py` rather than in a module of its own: the sweep
cut is what makes the upper bound constructive, and it shares the conductance
machinery with the clustering quality metrics.

Verification is a root-level script per module (`verify_spectrum.py`,
`verify_fiedler.py`, `verify_embedding.py`, `verify_clustering.py`), plus
`tests/test_spectral_graph.py` for the closed-form spectra and
`examples/spectral_graph_demo.py` for an end-to-end pass over the karate club.
Nothing inside `spectral_graph/` is meant to be executed directly — the package
imports itself absolutely, so it must be imported with the project root on
`sys.path`.

### Dependencies

Required:
- numpy
- scipy
- networkx (for graph construction and validation only)

Optional:
- matplotlib (for demo plotting)

k-means is implemented directly in NumPy (k-means++ seeding, Lloyd iterations,
seeded restarts) inside `clustering.py`, so scikit-learn is *not* a dependency
of the package. The scikit-learn notes in Part 2 remain as ecosystem context.

### Design Principles

1. **Direct implementation** - We compute Laplacians and eigendecompositions ourselves using NumPy/SciPy
2. **Sparse support** - Use `scipy.sparse` and `eigsh` for large graphs
3. **Validation** - Cross-check against NetworkX where available
4. **Educational** - Clear docstrings explaining the mathematics

---

## References

1. Fiedler, M. (1973). "Algebraic Connectivity of Graphs". Czechoslovak Mathematical Journal, 23(98), 298-305. https://dml.cz/handle/10338.dmlcz/101162

2. von Luxburg, U. (2007). "A Tutorial on Spectral Clustering". Statistics and Computing, 17(4), 395-416. https://link.springer.com/article/10.1007/s11222-007-9073-2

3. Ng, A., Jordan, M., Weiss, Y. (2002). "On Spectral Clustering: Analysis and an Algorithm". NIPS 14. https://papers.nips.cc/paper/2001/hash/801272ee79cfde795eef78aed585cc15-Abstract.html

4. Alon, N., Milman, V. (1985). "λ₁, Isoperimetric inequalities for graphs, and superconcentrators". Journal of Combinatorial Theory, Series B, 38(1), 73-88. https://www.sciencedirect.com/science/article/pii/0095895685900929

5. Chung, F. R. K. (1997). "Spectral Graph Theory". CBMS Regional Conference Series in Mathematics, 92. AMS. https://www.ams.org/books/cbms/092

6. SciPy Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

7. NetworkX Documentation: https://networkx.org/documentation/stable/reference/linalg.html

8. Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

9. Spielman, D. A. "Spectral and Algebraic Graph Theory" (draft textbook, Yale). The standard modern treatment of the Laplacian quadratic form, Cheeger's inequality, and effective resistance. http://www.cs.yale.edu/homes/spielman/sagt/sagt.pdf — course page: https://www.cs.yale.edu/homes/spielman/561/

10. Trevisan, L. "Lecture Notes on Expansion, Sparsest Cut, and Spectral Graph Theory" (2016). Proves both directions of Cheeger's inequality constructively, including the sweep-cut analysis this package's `sweep_cut` implements. https://lucatrevisan.github.io/books/expanders-2016.pdf

11. SciPy `eigsh` / ARPACK reference for the sparse Lanczos path: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
