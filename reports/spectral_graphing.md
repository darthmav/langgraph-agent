# Spectral Graph Theory: Discoveries Applied to Python

Each discovery below pairs a brief mathematical statement with a numpy/scipy/networkx
snippet. **Every snippet in this report was executed successfully in this environment**
(numpy 2.5.2, scipy 1.18.1, networkx 3.6.1, scikit-learn 1.9.0) before being included.

> Note on eigensolvers: for small graphs we use dense `scipy.linalg.eigh`. Sparse
> `scipy.sparse.linalg.eigsh` is reserved for large sparse matrices with `k < N-1`
> and an explicit `which=` parameter.

## 1. Laplacian Eigenstructure

**Math.** For an undirected graph, the Laplacian `L = D - A` is symmetric positive
semidefinite, so its eigenvalues are real and nonnegative:
`0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ`. The eigenvector of `λ₁ = 0` is the constant vector, and the
number of zero eigenvalues equals the number of connected components — hence a graph
is connected iff `λ₂ > 0`. `λ₂` is the **algebraic connectivity** (Fiedler value).

```python
import networkx as nx
import numpy as np
from scipy.linalg import eigh

G = nx.karate_club_graph()
L = nx.laplacian_matrix(G).astype(float).toarray()

# Dense symmetric eigensolver: all eigenvalues of L = D - A.
eigenvalues, eigenvectors = eigh(L)

print("5 smallest eigenvalues:", np.round(eigenvalues[:5], 4))
print("lambda_2 (Fiedler value):", round(float(eigenvalues[1]), 4))
print("lambda_max:", round(float(eigenvalues[-1]), 4))

assert abs(eigenvalues[0]) < 1e-8                  # smallest eigenvalue is 0
assert eigenvalues[1] > 0                          # connected => lambda_2 > 0
const = eigenvectors[:, 0]
assert np.allclose(const, const[0], atol=1e-6)     # 0-eigenvector is constant
```

**Verified output** (Zachary karate club, 34 nodes):

```
5 smallest eigenvalues: [-0.      1.1871  2.3943  2.9318  2.9683]
lambda_2 (Fiedler value): 1.1871
lambda_max: 52.0653
```

The run confirms `λ₁ = 0` with a constant eigenvector and `λ₂ = 1.1871 > 0`,
exactly as the theory predicts for a connected graph.

## 2. Fiedler Partitioning

**Math.** The Fiedler vector `v₂` (eigenvector of `λ₂`) minimizes the Rayleigh
quotient `x^T L x = ½ Σᵢⱼ Aᵢⱼ (xᵢ − xⱼ)²` over unit vectors orthogonal to the
constant vector: it is the smoothest non-constant labelling of the nodes, so its
entries are nearly constant inside a dense cluster and change sign across the
sparsest cut. The **sign cut** `S = {i : v₂(i) ≥ 0}` is therefore an approximate
minimum-ratio bipartition, and a small `λ₂` certifies that a bottleneck exists.

```python
"""Discovery 2: Fiedler-vector sign bipartition.

On success, appends its own section (math + code + captured output) to
reports/spectral_graphing.md, replacing the '_(pending)_' placeholder
under '## 2. Fiedler Partitioning'.
"""
import networkx as nx
import numpy as np
from scipy.linalg import eigh

lines = []
log = lines.append

# Two weakly-coupled clusters: 15+15 nodes, dense inside (p_in=0.6),
# sparse between (p_out=0.02). Scan seeds until the graph is connected --
# a disconnected graph has lambda_2 = 0 and the Fiedler cut is trivial.
seed = -1
G = None
for cand in range(500):
    trial = nx.planted_partition_graph(2, 15, 0.6, 0.02, seed=cand)
    if nx.is_connected(trial):
        seed, G = cand, trial
        break
assert G is not None, "no connected planted-partition graph found"

L = nx.laplacian_matrix(G).astype(float).toarray()

# Dense symmetric eigensolver for the small Laplacian.
eigenvalues, eigenvectors = eigh(L)
fiedler_value = float(eigenvalues[1])
lambda_3 = float(eigenvalues[2])
fiedler_vector = eigenvectors[:, 1]

# The sign of the Fiedler vector defines the bipartition.
side_a = {i for i, x in enumerate(fiedler_vector) if x >= 0}
side_b = set(G.nodes) - side_a

true_side = set(range(15))
agree = max(len(side_a & true_side), len(side_a - true_side))

log(f"connected planted partition found at seed {seed}")
log(f"Fiedler value lambda_2 = {fiedler_value:.4f}")
log(f"next eigenvalue lambda_3 = {lambda_3:.4f}")
log(f"partition sizes: {len(side_a)} / {len(side_b)}")
log(f"planted side-0 nodes recovered by the sign cut: {agree} / 15")

assert 0.05 < fiedler_value < 2.0   # weakly coupled: small but positive
assert agree >= 13                  # sign cut recovers the planted partition
log("OK: Fiedler sign cut recovers the planted bipartition")

output = "\n".join(lines)
print(output)
```

**Verified output** (planted partition, 2×15 nodes, p_in=0.6, p_out=0.02):

```
connected planted partition found at seed 0
Fiedler value lambda_2 = 0.4355
next eigenvalue lambda_3 = 3.7437
partition sizes: 15 / 15
planted side-0 nodes recovered by the sign cut: 15 / 15
OK: Fiedler sign cut recovers the planted bipartition
```

The sign cut isolates one planted community (15/15 nodes on the recovered
side), and the spectral gap `λ₃ − λ₂ = 3.3081` shows the
graph is essentially two clusters loosely stitched together — exactly the regime
where spectral partitioning works.

## 3. Cheeger's Inequality

**Math.** The conductance of a cut `S` is `h(S) = |∂S| / min(vol S, vol V\S)`,
and the graph conductance is `h_G = min_S h(S)`. Cheeger's inequality ties this
combinatorial bottleneck to the second eigenvalue `μ₂` of the **normalized**
Laplacian `L_norm = D^{-1/2} L D^{-1/2}`:
`μ₂/2 ≤ h_G ≤ √(2μ₂)`. Moreover, the upper bound is constructive: sorting nodes
by `D^{-1/2}u₂` and taking the best prefix (the **sweep cut**) yields a cut whose
conductance is at most `√(2μ₂)` — spectral partitioning with a guarantee.

```python
"""Discovery 3: Cheeger's inequality.

On success, appends its own section (math + code + captured output) to
reports/spectral_graphing.md, replacing the '_(pending)_' placeholder
under "## 3. Cheeger's Inequality".
"""
import networkx as nx
import numpy as np
from scipy.linalg import eigh

lines = []
log = lines.append

# Two weakly-coupled clusters: a graph with a genuine bottleneck,
# so the conductance is small and Cheeger's bound is informative.
G = nx.planted_partition_graph(2, 15, 0.6, 0.02, seed=0)
assert nx.is_connected(G)
n = G.number_of_nodes()

L = nx.laplacian_matrix(G).astype(float).toarray()
deg = np.diag(L)                                   # degree vector
Dm12 = np.diag(1.0 / np.sqrt(deg))
Lnorm = Dm12 @ L @ Dm12                            # normalized Laplacian

# Dense symmetric eigensolver for the small normalized Laplacian.
mu, U = eigh(Lnorm)
mu2 = float(mu[1])                                 # 2nd smallest eigenvalue of L_norm
v = Dm12 @ U[:, 1]                                 # Fiedler vector in original scale

# Sweep cut: sort nodes by v, take the best prefix by conductance.
order = np.argsort(v)
A = nx.to_numpy_array(G)
vol_total = float(deg.sum())

def conductance(S):
    S = set(S)
    cut = sum(A[i, j] for i in S for j in set(range(n)) - S)
    vol_S = float(deg[list(S)].sum())
    return cut / min(vol_S, vol_total - vol_S)

best_phi, best_k = np.inf, 0
for k in range(1, n):
    phi = conductance(order[:k])
    if phi < best_phi:
        best_phi, best_k = phi, k

lower = mu2 / 2.0
upper = np.sqrt(2.0 * mu2)

log(f"mu_2 (normalized Laplacian) = {mu2:.4f}")
log(f"sweep cut found at k = {best_k} nodes")
log(f"conductance of sweep cut  h = {best_phi:.4f}")
log(f"Cheeger bounds: mu_2/2 = {lower:.4f}  <=  h  <=  sqrt(2*mu_2) = {upper:.4f}")

assert lower - 1e-9 <= best_phi <= upper + 1e-9    # inequality holds
assert best_k == 15                                # sweep recovers the planted cut
log("OK: sweep-cut conductance satisfies Cheeger's inequality")

output = "\n".join(lines)
print(output)
```

**Verified output** (planted partition, 2×15 nodes, p_in=0.6, p_out=0.02):

```
mu_2 (normalized Laplacian) = 0.0535
sweep cut found at k = 15 nodes
conductance of sweep cut  h = 0.0357
Cheeger bounds: mu_2/2 = 0.0267  <=  h  <=  sqrt(2*mu_2) = 0.3270
OK: sweep-cut conductance satisfies Cheeger's inequality
```

The sweep cut lands exactly on the planted bottleneck (k = 15), and its
conductance sits inside the Cheeger window `[μ₂/2, √(2μ₂)]` — the eigenvalue
both certifies the bottleneck and constructs the cut.

## 4. Spectral Clustering

**Math.** For `k` weakly-coupled clusters, the normalized Laplacian has `k` small
eigenvalues before a spectral gap (`μ₁ ≈ … ≈ μ_k ≈ 0 < μ_{k+1}`). Stacking the
first `k` eigenvectors as columns gives an `n × k` **spectral embedding** in which
nodes from the same cluster map to nearly the same point (the embedding is a
perturbed indicator vector per cluster). Running plain **k-means** on the
row-normalized embedding (Ng–Jordan–Weiss) therefore recovers the communities —
the number of clusters `k` can be read off the eigengap.

```python
"""Discovery 4: Spectral clustering (k-way via eigenvector embedding + k-means).

On success, appends its own section (math + code + captured output) to
reports/spectral_graphing.md, replacing the '_(pending)_' placeholder
under '## 4. Spectral Clustering'.
"""
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.cluster.vq import kmeans2

lines = []
log = lines.append

# Three weakly-coupled clusters of 12 nodes each. Scan seeds until the
# graph is connected -- a disconnected graph has mu_2 = 0 and the
# eigengap test below would be meaningless.
k = 3
seed = -1
G = None
for cand in range(500):
    trial = nx.planted_partition_graph(k, 12, 0.6, 0.02, seed=cand)
    if nx.is_connected(trial):
        seed, G = cand, trial
        break
assert G is not None, "no connected planted-partition graph found"
n = G.number_of_nodes()

L = nx.laplacian_matrix(G).astype(float).toarray()
deg = np.diag(L)
Dm12 = np.diag(1.0 / np.sqrt(deg))
Lnorm = Dm12 @ L @ Dm12                            # normalized Laplacian

# Dense symmetric eigensolver; embed nodes using the first k eigenvectors.
eigenvalues, U = eigh(Lnorm)
X = U[:, :k]                                       # n x k spectral embedding
X = X / np.linalg.norm(X, axis=1, keepdims=True)   # row-normalize (Ng-Jordan-Weiss)

# k-means on the embedding rows (scipy only, fixed seed for reproducibility).
centroids, labels = kmeans2(X, k, seed=0, minit="++")

true_labels = np.repeat(range(k), 12)
# Best permutation match between found and true labels.
from itertools import permutations
best = max(sum(labels[i] == p[true_labels[i]] for i in range(n))
           for p in permutations(range(k)))

log(f"connected planted partition found at seed {seed}")
log(f"eigenvalues mu_1..mu_5: {np.round(eigenvalues[:5], 4)}")
log(f"spectral gap mu_4 - mu_3 = {eigenvalues[3] - eigenvalues[2]:.4f}")
log(f"cluster sizes found: {np.bincount(labels).tolist()}")
log(f"nodes matching planted communities (best permutation): {best} / {n}")

assert eigenvalues[2] < 0.2 < eigenvalues[3]       # 3 near-zero eigenvalues, then a gap
assert best >= n - 2                               # clustering recovers the communities
log("OK: k-means on the spectral embedding recovers the planted communities")

output = "\n".join(lines)
print(output)
```

**Verified output** (planted partition, 3×12 nodes, p_in=0.6, p_out=0.02):

```
connected planted partition found at seed 0
eigenvalues mu_1..mu_5: [0.     0.0681 0.0932 0.6205 0.6544]
spectral gap mu_4 - mu_3 = 0.5273
cluster sizes found: [12, 12, 12]
nodes matching planted communities (best permutation): 36 / 36
OK: k-means on the spectral embedding recovers the planted communities
```

Three near-zero eigenvalues followed by a clear gap announce `k = 3`, and
k-means on the embedding recovers the planted communities essentially exactly —
no graph-specific clustering code required.

## 5. Spectral Sparsification

_(pending)_

## 6. Graph Fourier Transform

_(pending)_
