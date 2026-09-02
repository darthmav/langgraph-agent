# Spectral Mathematics Overview

## Introduction

Spectral mathematics is a branch of mathematics that studies the properties of linear operators through their eigenvalues and eigenvectors. The term "spectral" refers to the spectrum of eigenvalues associated with a matrix or operator.

## Core Concepts

### Eigenvalues and Eigenvectors

For a square matrix $A \in \mathbb{R}^{n \times n}$, an eigenvalue $\lambda$ and corresponding eigenvector $v$ satisfy:

$$Av = \lambda v$$

where $v \neq 0$.

### Spectral Decomposition

For a symmetric matrix $A$, the spectral decomposition (eigendecomposition) is:

$$A = Q\Lambda Q^T$$

where:
- $Q$ is an orthogonal matrix whose columns are eigenvectors
- $\Lambda$ is a diagonal matrix of eigenvalues

### Key Algorithms

1. **Power Iteration**: Finds the dominant eigenvalue
   - Complexity: $O(n^2)$ per iteration
   - Converges to largest eigenvalue in magnitude

2. **Lanczos Algorithm**: Efficient for sparse symmetric matrices
   - Reduces matrix to tridiagonal form
   - Complexity: $O(nk)$ for $k$ iterations

3. **QR Algorithm**: Computes all eigenvalues
   - Standard method in numpy.linalg.eig

## Subdomains

### 1. Spectral Graph Theory
Studies graphs through eigenvalues of associated matrices (adjacency, Laplacian).

### 2. Spectral Signal Processing
Analyzes signals in the frequency domain using Fourier transforms.

### 3. Spectral Methods for PDEs
Solves differential equations using basis function expansions.

## Python Ecosystem

- **numpy.linalg**: Basic linear algebra (eig, svd, eigh)
- **scipy.linalg**: Advanced linear algebra routines
- **scipy.sparse**: Sparse matrix operations
- **scipy.signal**: Signal processing (FFT, filters)
- **matplotlib**: Visualization

## References

1. Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
2. Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
3. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*. Johns Hopkins University Press.
