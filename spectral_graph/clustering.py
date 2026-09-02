"""
Spectral clustering, conductance, and Cheeger bounds.

Provides k-way spectral clustering built on the Laplacian eigenvector
embedding, plus the conductance machinery that makes a partition's quality
measurable: the sweep cut over the Fiedler vector and the two Cheeger bounds
that bracket the true conductance of the graph.

k-means is implemented here in NumPy (k-means++ seeding, Lloyd iterations)
rather than imported, so the package depends on nothing beyond numpy, scipy,
and networkx.
"""

import networkx as nx
import numpy as np

from spectral_graph.embedding import spectral_embedding
from spectral_graph.spectrum import compute_spectrum


def _kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-8,
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster rows of X into k groups with Lloyd's algorithm and k-means++ seeding.

    Restarts `n_init` times from different seeds and keeps the run with the
    lowest inertia. Deterministic for a fixed `random_state`.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features)
    k : int
        Number of clusters
    n_init : int, default 10
        Number of restarts
    max_iter : int, default 300
        Maximum Lloyd iterations per restart
    tol : float, default 1e-8
        Centroid movement below which a run is considered converged
    random_state : int, default 0
        Seed for the restart RNG

    Returns
    -------
    numpy.ndarray
        Integer labels of shape (n_samples,)
    """
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    best_labels = np.zeros(n, dtype=int)
    best_inertia = np.inf

    for _ in range(n_init):
        # --- k-means++ seeding ---
        centers = np.empty((k, X.shape[1]), dtype=float)
        centers[0] = X[rng.integers(n)]
        closest = np.sum((X - centers[0]) ** 2, axis=1)
        for j in range(1, k):
            total = closest.sum()
            if total <= 0:
                # All points already coincide with a center; pick uniformly.
                centers[j] = X[rng.integers(n)]
            else:
                centers[j] = X[rng.choice(n, p=closest / total)]
            closest = np.minimum(closest, np.sum((X - centers[j]) ** 2, axis=1))

        # --- Lloyd iterations ---
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # (n, k) squared distances to every center
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = centers.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    new_centers[j] = members.mean(axis=0)
                # An empty cluster keeps its center rather than collapsing.

            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if shift <= tol:
                break

        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.min(dists, axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels


def spectral_clustering(
    G: nx.Graph,
    k: int = 2,
    normalized: bool = True,
    random_state: int = 0,
    n_init: int = 10,
) -> np.ndarray:
    """
    Partition a graph into k clusters via its Laplacian eigenvectors.

    Embeds the nodes using the k eigenvectors of the smallest eigenvalues
    (including the trivial one) and runs k-means in that space. With
    `normalized=True` the rows of the embedding are scaled to unit length
    first, which is the Ng-Jordan-Weiss formulation.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    k : int, default 2
        Number of clusters
    normalized : bool, default True
        If True, use the normalized Laplacian and row-normalize the embedding
    random_state : int, default 0
        Seed for k-means, so repeated calls give the same labels
    n_init : int, default 10
        Number of k-means restarts

    Returns
    -------
    numpy.ndarray
        Integer labels of shape (n_nodes,), ordered as `list(G.nodes())`

    Raises
    ------
    ValueError
        If k < 1 or k exceeds the number of nodes

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import spectral_clustering
    >>> G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    >>> G.add_edge(0, 5)
    >>> labels = spectral_clustering(G, k=2)
    >>> labels.shape
    (10,)
    >>> # The two cliques land in different clusters
    >>> len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
    True
    """
    n = G.number_of_nodes()

    if k < 1:
        raise ValueError("Number of clusters must be at least 1")
    if k > n:
        raise ValueError(f"Cannot form {k} clusters from a graph with {n} nodes")

    if k == 1:
        return np.zeros(n, dtype=int)

    # k smallest eigenvectors, trivial one included (Shi-Malik / NJW).
    X = spectral_embedding(G, dim=k, normalized=normalized, use_fiedler=False)

    if normalized:
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-10, row_norms, 1.0)
        X = X / row_norms

    return _kmeans(X, k, n_init=n_init, random_state=random_state)


def conductance(G: nx.Graph, S) -> float:
    """
    Compute the conductance of a node set S.

    phi(S) = w(S, V\\S) / min(vol(S), vol(V\\S)), where vol(S) is the sum of
    the degrees of S. Edge weights are honoured; an unweighted graph counts
    each edge as 1.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    S : iterable
        Nodes on one side of the cut

    Returns
    -------
    float
        Conductance of the cut, in [0, 1]. Returns infinity for an empty
        or whole-graph S, which has no finite conductance.

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import conductance
    >>> G = nx.path_graph(4)
    >>> # Cutting P_4 in the middle: 1 crossing edge, vol = 3 on each side
    >>> round(conductance(G, {0, 1}), 6)
    0.333333
    """
    S = set(S)
    if not S or len(S) == G.number_of_nodes():
        return float("inf")

    boundary = 0.0
    vol_S = 0.0
    for u in S:
        for v, data in G[u].items():
            w = data.get("weight", 1.0)
            vol_S += w
            if v not in S:
                boundary += w

    total_vol = sum(w for _, w in G.degree(weight="weight"))
    denom = min(vol_S, total_vol - vol_S)
    if denom <= 0:
        return float("inf")

    return boundary / denom


def sweep_cut(G: nx.Graph, normalized: bool = True) -> tuple:
    """
    Find the best sweep cut along the Fiedler vector.

    Orders the nodes by their Fiedler entry and evaluates every prefix as a
    candidate cut, returning the prefix of lowest conductance. This is the
    constructive half of the Cheeger inequality: the cut it finds is
    guaranteed to satisfy phi <= sqrt(2 * lambda_2).

    Parameters
    ----------
    G : networkx.Graph
        Input graph (must have at least 2 nodes)
    normalized : bool, default True
        If True, sweep the normalized Laplacian's Fiedler vector rescaled by
        D^(-1/2), which is the vector the Cheeger bound is stated for

    Returns
    -------
    tuple
        (best_set, best_conductance) - the node set of the winning prefix and
        its conductance

    Raises
    ------
    ValueError
        If the graph has fewer than 2 nodes

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import sweep_cut
    >>> G = nx.barbell_graph(5, 0)
    >>> S, phi = sweep_cut(G)
    >>> sorted(S) in ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    True
    """
    n = G.number_of_nodes()
    if n < 2:
        raise ValueError("Graph must have at least 2 nodes")

    # spectral_embedding with dim=1 returns the Fiedler vector as a column.
    vec = spectral_embedding(G, dim=1, normalized=normalized, use_fiedler=True)[:, 0]

    nodes = list(G.nodes())
    if normalized:
        degrees = np.array([d for _, d in G.degree(weight="weight")], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = 1.0 / np.sqrt(degrees)
        scale[~np.isfinite(scale)] = 0.0
        vec = vec * scale

    order = np.argsort(vec)

    total_vol = sum(w for _, w in G.degree(weight="weight"))
    in_S: set = set()
    boundary = 0.0
    vol_S = 0.0
    best_set: set = set()
    best_phi = float("inf")

    # Every prefix but the last; the whole graph is not a cut.
    for idx in order[:-1]:
        u = nodes[idx]
        for v, data in G[u].items():
            w = data.get("weight", 1.0)
            vol_S += w
            # An edge to a node already inside stops crossing the cut.
            boundary += -w if v in in_S else w
        in_S.add(u)

        denom = min(vol_S, total_vol - vol_S)
        if denom <= 0:
            continue
        phi = boundary / denom
        if phi < best_phi:
            best_phi = phi
            best_set = set(in_S)

    return best_set, best_phi


def cheeger_bounds(G: nx.Graph) -> tuple:
    """
    Compute the Cheeger bounds on the conductance of a graph.

    The Cheeger inequality brackets the graph's conductance h_G between the
    second eigenvalue of the normalized Laplacian and its square root:

        lambda_2 / 2 <= h_G <= sqrt(2 * lambda_2)

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    Returns
    -------
    tuple
        (lower, upper) bounds on the conductance

    Examples
    --------
    >>> import networkx as nx
    >>> from spectral_graph import cheeger_bounds, sweep_cut
    >>> G = nx.barbell_graph(6, 0)
    >>> lower, upper = cheeger_bounds(G)
    >>> _, phi = sweep_cut(G)
    >>> lower <= phi <= upper
    True
    """
    eigenvalues = compute_spectrum(G, k=2, normalized=True, which="SM")
    if len(eigenvalues) < 2:
        raise ValueError("Graph too small to compute Cheeger bounds")

    lambda2 = float(eigenvalues[1])
    lambda2 = max(lambda2, 0.0)  # clamp solver noise around 0

    return lambda2 / 2.0, float(np.sqrt(2.0 * lambda2))


if __name__ == "__main__":
    # This script should not be run directly from within spectral_graph/
    # Run via: python verify_clustering.py from project root
    print("Do not run clustering.py directly. Use verify_clustering.py from project root.")
