#!/usr/bin/env python3
"""
Demo of the spectral_graph package on Zachary's karate club.

Walks the whole pipeline on one real graph: Laplacian construction, the
spectrum, the Fiedler vector and the bipartition it induces, the sweep cut and
its Cheeger bracket, k-way spectral clustering, and a 2-D embedding. The known
faction split of the club is used as ground truth throughout.

Run from the project root:

    python examples/spectral_graph_demo.py
"""

import os
import sys

# The package is not installed; import it from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import networkx as nx
import numpy as np

from spectral_graph import (
    algebraic_connectivity,
    cheeger_bounds,
    compute_spectrum,
    conductance,
    fiedler_partition,
    fiedler_vector,
    laplacian_matrix,
    spectral_bipartition,
    spectral_clustering,
    spectral_embedding,
    sweep_cut,
)


def ground_truth(G):
    """The two real factions of the karate club, as sets of nodes."""
    mr_hi = {n for n, d in G.nodes(data=True) if d["club"] == "Mr. Hi"}
    return mr_hi, set(G.nodes()) - mr_hi


def agreement(partition, truth):
    """Fraction of nodes a 2-set partition places on the right side.

    The labels of a spectral partition are arbitrary, so both matchings are
    tried and the better one is reported.
    """
    (a, b), (x, y) = partition, truth
    n = len(x) + len(y)
    return max(len(a & x) + len(b & y), len(a & y) + len(b & x)) / n


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def main():
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    truth = ground_truth(G)

    print("Zachary's karate club")
    print(f"  {n} nodes, {G.number_of_edges()} edges "
          f"(weighted: interaction counts)")
    print(f"  known factions: {len(truth[0])} with Mr. Hi, {len(truth[1])} with the Officer")

    # --- Laplacian -----------------------------------------------------
    section("1. Laplacian")
    L = laplacian_matrix(G)
    print(f"  L = D - A is {L.shape[0]}x{L.shape[1]}, {L.nnz} stored entries "
          f"({100 * L.nnz / n**2:.1f}% dense)")
    residual = float(np.linalg.norm(L @ np.ones(n)))
    print(f"  ||L @ 1|| = {residual:.2e}  (the constant vector spans the null space)")
    assert residual < 1e-9

    # --- Spectrum ------------------------------------------------------
    section("2. Spectrum")
    eigenvalues = compute_spectrum(G)
    print(f"  6 smallest: {np.round(eigenvalues[:6], 4)}")
    print(f"  largest:    {eigenvalues[-1]:.4f}")
    zeros = int(np.sum(np.abs(eigenvalues) < 1e-8))
    print(f"  {zeros} zero eigenvalue -> {nx.number_connected_components(G)} connected component")
    assert zeros == nx.number_connected_components(G)

    normalized = compute_spectrum(G, normalized=True)
    print(f"  normalized spectrum spans [{normalized.min():.4f}, {normalized.max():.4f}] "
          f"(theory: [0, 2])")
    assert -1e-9 < normalized.min() and normalized.max() < 2 + 1e-9

    # --- Fiedler vector ------------------------------------------------
    section("3. Fiedler vector and bipartition")
    lambda2 = algebraic_connectivity(G)
    v = fiedler_vector(G)
    print(f"  algebraic connectivity lambda_2 = {lambda2:.4f} (> 0, so connected)")
    print(f"  NetworkX agrees: {nx.algebraic_connectivity(G, weight='weight'):.4f}")
    print(f"  Fiedler vector: sum {np.sum(v):+.2e}, norm {np.linalg.norm(v):.4f}")

    result = spectral_bipartition(G)
    acc = agreement(fiedler_partition(G), truth)
    print(f"  sign cut: {len(result['set1'])} / {len(result['set2'])} nodes, "
          f"{result['cut_size']} edges cut, balance {result['balance']:.2f}")
    print(f"  agreement with the real faction split: {acc:.1%}")
    assert acc >= 0.85, f"sign cut recovered only {acc:.1%} of the split"

    # --- Sweep cut and Cheeger ------------------------------------------
    section("4. Sweep cut and the Cheeger bracket")
    S, phi = sweep_cut(G)
    lower, upper = cheeger_bounds(G)
    print(f"  best sweep cut: {len(S)} nodes, conductance phi = {phi:.4f}")
    print(f"  conductance of the Fiedler sign cut: {conductance(G, result['set1']):.4f}")
    print(f"  Cheeger: {lower:.4f} <= phi <= {upper:.4f}  "
          f"(lambda_2/2 and sqrt(2*lambda_2), normalized Laplacian)")
    assert lower - 1e-9 <= phi <= upper + 1e-9

    # --- Spectral clustering --------------------------------------------
    section("5. Spectral clustering")
    labels2 = spectral_clustering(G, k=2)
    clusters2 = ({i for i in range(n) if labels2[i] == 0},
                 {i for i in range(n) if labels2[i] == 1})
    acc2 = agreement(clusters2, truth)
    print(f"  k=2: sizes {sorted(np.bincount(labels2).tolist(), reverse=True)}, "
          f"agreement with the faction split {acc2:.1%}")
    assert acc2 >= 0.85, f"k=2 clustering recovered only {acc2:.1%}"

    for k in (3, 4):
        labels = spectral_clustering(G, k=k)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        cuts = [conductance(G, {i for i in range(n) if labels[i] == c}) for c in range(k)]
        print(f"  k={k}: sizes {sizes}, per-cluster conductance "
              f"{[round(c, 3) for c in cuts]}")
        assert len(set(labels)) == k

    # --- Embedding --------------------------------------------------------
    section("6. Spectral embedding")
    X = spectral_embedding(G, dim=2)
    print(f"  2-D embedding: {X.shape[0]} points, columns orthonormal "
          f"({np.allclose(X.T @ X, np.eye(2), atol=1e-9)})")
    hi, officer = truth
    centroid_hi = X[sorted(hi)].mean(axis=0)
    centroid_off = X[sorted(officer)].mean(axis=0)
    separation = float(np.linalg.norm(centroid_hi - centroid_off))
    spread = float(np.linalg.norm(X - X.mean(axis=0), axis=1).mean())
    print(f"  faction centroids: Mr. Hi {np.round(centroid_hi, 3)}, "
          f"Officer {np.round(centroid_off, 3)}")
    print(f"  centroid separation {separation:.3f} vs mean spread {spread:.3f}")
    assert separation > spread / 2, "the factions did not separate in the embedding"

    print("\nAll demo stages ran and their assertions held.")


if __name__ == "__main__":
    main()
