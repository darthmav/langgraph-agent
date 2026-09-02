"""Runnable code examples accompanying docs/spectral_graphing_report.md.

Every snippet in the report lives here as a self-contained function using only
numpy / scipy / networkx / scikit-learn. Run directly to execute all examples:

    python examples/spectral_graphing_examples.py
"""

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def example_fiedler_partition():
    """Discovery 1: Fiedler vector sign bipartition of a two-cluster graph.

    The graph must be connected for this to mean anything. A disconnected one
    has lambda_2 = 0 with a null space spanned by the component indicators, so
    "the" Fiedler vector is not unique -- any vector in that space is a valid
    eigenvector and the sign cut it induces is arbitrary. At p_out = 0.02 a
    2x15 planted partition is often drawn with no cross edges at all, so the
    seed is scanned rather than fixed.
    """
    seed, G = -1, None
    for cand in range(500):
        trial = nx.planted_partition_graph(2, 15, 0.6, 0.02, seed=cand)
        if nx.is_connected(trial):
            seed, G = cand, trial
            break
    assert G is not None, "no connected planted-partition graph found"

    # Dense eigh, not sparse eigsh: this Laplacian is 30x30, and ARPACK's
    # which="SM" starts from a random vector, so it returns a different answer
    # run to run on a degenerate or near-degenerate low end.
    L = nx.laplacian_matrix(G).astype(float).toarray()
    eigenvalues, eigenvectors = eigh(L)
    fiedler_value = float(eigenvalues[1])
    fiedler_vector = eigenvectors[:, 1]

    cut = {i for i, x in enumerate(fiedler_vector) if x >= 0}
    true_side = set(range(15))
    agree = max(len(cut & true_side), len(cut - true_side))

    print(f"  connected planted partition found at seed {seed}")
    print(f"  Fiedler value lambda_2 = {fiedler_value:.4f}")
    print(f"  Sign cut recovered {agree}/15 nodes of a planted side")
    assert 0.05 < fiedler_value < 2.0  # weakly coupled: small but positive
    assert agree >= 13


def example_cheeger():
    """Discovery 2: Cheeger sweep cut on the Fiedler vector."""
    G = nx.planted_partition_graph(2, 20, 0.5, 0.02, seed=3)
    L = nx.laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=2, which="SM")
    lambda2 = vals[1]
    order = np.argsort(vecs[:, 1])
    best_phi, best_k = np.inf, 0
    for k in range(1, len(order)):
        S = set(order[:k])
        boundary = sum(1 for u, v in G.edges if (u in S) != (v in S))
        phi = boundary / min(len(S), G.number_of_nodes() - len(S))
        if phi < best_phi:
            best_phi, best_k = phi, k
    upper = np.sqrt(2 * lambda2)
    print(f"  lambda_2 = {lambda2:.4f}; sweep conductance = {best_phi:.4f} "
          f"(cut size {best_k}); Cheeger upper bound sqrt(2*lambda_2) = {upper:.4f}")
    assert best_phi <= upper + 1e-9


def example_spectral_clustering():
    """Discovery 3: spectral clustering via the k smallest Laplacian eigenvectors."""
    from sklearn.cluster import KMeans

    G = nx.planted_partition_graph(3, 20, 0.6, 0.02, seed=11)
    L = nx.laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=3, which="SM")
    labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(vecs)
    truth = np.repeat([0, 1, 2], 20)
    # cluster labels are arbitrary; measure best-permutation accuracy
    from itertools import permutations
    acc = max(np.mean(labels == np.array(p)[truth]) for p in permutations(range(3)))
    print(f"  smallest eigenvalues: {np.round(vals, 3)}")
    print(f"  spectral clustering accuracy vs planted blocks: {acc:.2%}")
    assert acc >= 0.9


def example_expander():
    """Discovery 4: Alon-Boppana / Ramanujan check on a random regular graph."""
    d = 4
    G = nx.random_regular_graph(d, 200, seed=5)
    A = nx.adjacency_matrix(G).astype(float)
    lam2 = float(eigsh(A, k=2, which="LA")[0][0])  # second-largest eigenvalue
    alon_boppana = 2 * np.sqrt(d - 1)
    print(f"  second adjacency eigenvalue lambda = {lam2:.3f}")
    print(f"  Alon-Boppana limit 2*sqrt(d-1) = {alon_boppana:.3f}; "
          f"Ramanujan would need lambda <= {alon_boppana:.3f}")
    assert lam2 > alon_boppana - 1.0  # near the limit, well below d


def example_sparsification():
    """Discovery 5: effective-resistance sampling preserves the Laplacian spectrum."""
    n = 60
    G = nx.gnm_random_graph(n, 4 * n, seed=2)
    L = nx.laplacian_matrix(G).astype(float).toarray()
    lam_max_full = float(np.linalg.eigvalsh(L)[-1])

    # effective resistance of every edge via the Laplacian pseudoinverse
    Lpinv = np.linalg.pinv(L)
    edges = list(G.edges)
    R = np.array([Lpinv[u, u] + Lpinv[v, v] - 2 * Lpinv[u, v] for u, v in edges])
    p = R / R.sum()

    rng = np.random.default_rng(0)
    q = 6 * n  # number of samples
    counts = rng.multinomial(q, p)
    Ls = np.zeros_like(L)
    for (u, v), c in zip(edges, counts, strict=True):
        if c:
            w = c / (q * p[edges.index((u, v))])
            Ls[u, u] += w
            Ls[v, v] += w
            Ls[u, v] -= w
            Ls[v, u] -= w
    lam_max_sparse = float(np.linalg.eigvalsh(Ls)[-1])
    kept = int(np.count_nonzero(counts))
    ratio = lam_max_sparse / lam_max_full
    print(f"  kept {kept} of {len(edges)} edges; "
          f"lambda_max ratio sparsifier/original = {ratio:.2f}")
    assert 0.3 < ratio < 3.0  # same order of magnitude with ~40% of edges


def example_gft():
    """Discovery 6: graph Fourier transform and low-pass filtering."""
    G = nx.grid_2d_graph(8, 8)
    G = nx.convert_node_labels_to_integers(G)
    L = nx.laplacian_matrix(G).astype(float).toarray()
    vals, U = np.linalg.eigh(L)
    coords = np.array([(i // 8, i % 8) for i in range(64)], dtype=float)
    x = np.sin(coords[:, 0] / 8 * np.pi) + 0.3 * np.random.default_rng(1).normal(size=64)
    xhat = U.T @ x                      # graph Fourier transform
    xhat_lp = np.where(vals <= 1.0, xhat, 0.0)  # ideal low-pass
    x_smooth = U @ xhat_lp              # inverse GFT
    residual = np.linalg.norm(x - x_smooth) / np.linalg.norm(x)
    energy_low = (xhat_lp**2).sum() / (xhat**2).sum()
    print(f"  {energy_low:.1%} of signal energy sits at eigenvalues <= 1; "
          f"low-pass relative residual = {residual:.2f}")
    assert energy_low > 0.8


def example_heat_diffusion():
    """Discovery 7: heat-kernel smoothing via the Laplacian exponential."""
    from scipy.linalg import expm

    G = nx.karate_club_graph()
    L = nx.laplacian_matrix(G).astype(float).toarray()
    x0 = np.zeros(G.number_of_nodes())
    x0[0] = 1.0  # unit impulse at node 0
    for t in (0.1, 1.0, 5.0):
        xt = expm(-t * L) @ x0
        spread = np.count_nonzero(xt > 1e-3)
        print(f"  t={t:>4}: mass on {spread:>2} nodes, max value {xt.max():.3f}")
    xt = expm(-50.0 * L) @ x0
    assert np.allclose(xt, xt.mean(), atol=1e-2)  # converged to average


def example_cheb_conv():
    """Discovery 8: Chebyshev polynomial spectral convolution (ChebNet layer)."""
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    L = sparse.csgraph.laplacian(nx.adjacency_matrix(G).astype(float), normed=True)
    lam_max = float(eigsh(L, k=1, which="LA", return_eigenvectors=False)[0])
    L_tilde = (2.0 / lam_max) * L - sparse.eye(n)  # rescale spectrum to [-1, 1]

    rng = np.random.default_rng(4)
    X = rng.normal(size=(n, 5))          # 5 input features per node
    W = rng.normal(size=(5, 3))          # 3 output features
    K = 3                                # Chebyshev order = K-hop support
    T_prev, T_curr = X, L_tilde @ X      # T_0, T_1
    out = T_prev + T_curr                # theta_0 = theta_1 = 1 for demo
    for _ in range(2, K):
        T_prev, T_curr = T_curr, 2 * (L_tilde @ T_curr) - T_prev
        out = out + T_curr
    Z = out @ W
    print(f"  ChebNet layer: input {X.shape} -> output {Z.shape}, "
          f"K={K}-hop localized, no eigendecomposition used")
    assert Z.shape == (n, 3)


def example_lanczos_scaling():
    """Discovery 9: eigsh/Lanczos scales to graphs where dense eigh cannot."""
    import time

    G = nx.barabasi_albert_graph(5000, 3, seed=8)
    L = nx.laplacian_matrix(G).astype(float)  # sparse
    t0 = time.perf_counter()
    vals = eigsh(L, k=6, which="SM")
    dt = time.perf_counter() - t0
    print(f"  6 smallest eigenvalues of a 5000-node sparse Laplacian "
          f"in {dt:.2f}s: {np.round(vals[0], 3)}")
    assert dt < 30


def example_modularity():
    """Discovery 10: spectral modularity maximization with the leading eigenpair."""
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).astype(float).toarray()
    k = A.sum(axis=1)
    m2 = k.sum()  # 2m
    B = A - np.outer(k, k) / m2
    vals, vecs = np.linalg.eigh(B)
    s = np.sign(vecs[:, -1])
    Q = float(s @ B @ s / (2 * m2))
    print(f"  leading modularity eigenvalue = {vals[-1]:.3f}; "
          f"sign split gives Q = {Q:.3f}")
    assert Q > 0.3  # karate club splits well into two factions


EXAMPLES = [
    ("Fiedler value & spectral bipartition", example_fiedler_partition),
    ("Cheeger inequality sweep cut", example_cheeger),
    ("Spectral clustering", example_spectral_clustering),
    ("Expanders & Alon-Boppana", example_expander),
    ("Spectral sparsification", example_sparsification),
    ("Graph Fourier transform", example_gft),
    ("Heat-kernel diffusion", example_heat_diffusion),
    ("Chebyshev spectral convolution (ChebNet)", example_cheb_conv),
    ("Lanczos scaling with eigsh", example_lanczos_scaling),
    ("Spectral modularity maximization", example_modularity),
]


def main():
    for title, fn in EXAMPLES:
        print(f"[{title}]")
        fn()
        print()
    print("All spectral graphing examples ran successfully.")


if __name__ == "__main__":
    main()
