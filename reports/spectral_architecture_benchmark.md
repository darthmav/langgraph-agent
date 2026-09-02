# What Spectral Methods Buy You, Per Architecture

A measured assessment of the `spectral_graph` package: an implementation review,
and a benchmark across 14 graph architectures that answers a question the
existing reports do not — not "is this correct?" but "when is it worth
reaching for?"

Reproduce with `python scripts/spectral_benchmark.py`. Every number below is
from that script on this machine (numpy 2.5.2, scipy 1.18.1, networkx 3.6.1,
Python 3.14.7). Seeded throughout.

The distinction from `reports/spectral_graphing.md` and
`reports/spectral_conclusion.md` matters: those establish that the algorithms
compute the right thing, on graphs chosen to show them working. Neither
compares against a non-spectral baseline, so neither can say whether the
spectrum earned anything. A method that scores 1.000 on a planted partition
that greedy modularity also scores 1.000 on has demonstrated correctness and
no advantage.

---

## 1. Implementation review

The package is in good shape. 80 tests pass, the Laplacians match NetworkX on
weighted and self-looped graphs, the directed-graph refusal from commit
`50d112a` holds at every public entry point, and the k-means is a correct
k-means++ / Lloyd implementation. Two defects were found and fixed; both are
now pinned by tests (87 pass).

### 1.1 `sweep_cut` disagreed with `conductance` on its own returned set

`sweep_cut` returns `(S, phi)` where `phi` is the conductance of `S` — so it is
by definition the number `conductance(G, S)` computes. On a graph with a
self-loop the two disagreed by 91%:

```
barbell_graph(5, 0) + self-loop at node 0
  sweep_cut  -> phi = 0.090909
  conductance(G, same set) -> 0.047619
```

Two compounding causes, both the self-loop bug the Laplacian had already been
audited for:

1. **The loop was charged to the cut.** The boundary update reads
   `boundary += -w if v in in_S else w`, and `u` is added to `in_S` only
   *after* the loop over its neighbours — so when `v == u`, `u` is not yet
   inside and its own self-loop was counted as an edge leaving `S`. A
   self-loop has both ends on the same side and can never cross a cut.
2. **The two halves of the ratio used different degree conventions.**
   `total_vol` came from `G.degree(weight="weight")`, which counts a self-loop
   twice; the running `vol_S` accumulates from `G[u]`, which reaches it once —
   the same mismatch `conductance` was already fixed for, with the fix's
   comment sitting one function above the code that still had it.

This is not only arithmetic. `sweep_cut` *minimises* the number it reports, so
a wrong objective can select a different prefix; and the `phi` it returns is
what gets checked against `cheeger_bounds`. Both degree reads now go through
`_degree_vector`, so the three conductance paths cannot drift apart again.

### 1.2 Every sparse branch used `eigsh(..., which="SM")`

Four call sites — `compute_spectrum`, `compute_eigenpairs`,
`spectral_embedding`, `fiedler_vector` — reached ARPACK's `SM` mode for any
graph with n ≥ 50. That mode iterates on `L` directly, and its convergence is
governed by the *relative* separation of the eigenvalues being chased. On a
graph with a bottleneck that separation is exactly what is tiny, so the mode is
slowest on precisely the graphs a spectral analysis exists to study:

| graph | λ₂ | `SM` | shift-invert | slower by |
|---|---|---|---|---|
| path (n=1200) | 6.9e-06 | 6.34 s | 0.002 s | **2968×** |
| cycle (n=1200) | 2.7e-05 | 3.03 s | 0.002 s | **1435×** |
| barbell (n=1000) | 2.3e-05 | 8.65 s | 0.017 s | **499×** |
| grid 40×40 | 6.2e-03 | 0.050 s | 0.009 s | 5.4× |
| random 6-regular (n=1200) | 1.54 | 0.009 s | 0.135 s | 0.07× |

Worse than slow: **it does not always finish.** ARPACK draws a random starting
residual unless given one, and on `path_1200` the `SM` call raises
`ArpackNoConvergence` in roughly 1 run in 6, after ~6 s of work — an
intermittent exception rather than a slow answer. Shift-invert failed 0/6 on
the same graphs, and is if anything the more accurate of the two (relative
error 4.4e-09 vs 2.8e-08 against a `tol=0` reference on the barbell).

Fixed by routing all four sites through `spectral_graph.spectrum.smallest_eigsh`,
which factorizes `L - σI` once and iterates on the inverse. `σ` is a hair below
zero, scaled by the largest diagonal entry — `σ = 0` asks SuperLU to factorize a
singular matrix, since 0 is an eigenvalue of every Laplacian. It falls back to
the old call if the factorization fails, so nothing is worse off than before.

The trade is real and bounded the right way: on an expander, where λ₂ = O(1),
the factorization is pure overhead and this is ~15× *slower*. Losing a hundred
milliseconds on the easy case to win six seconds — and an intermittent
exception — on the hard one is the trade worth making, because the hard case is
the one that scales into a hang.

### 1.3 The test suite ran entirely below the threshold it was guarding

All 80 tests completed in 0.48 s, which is the tell: every graph in the suite
had n < 50, so the dense branch was fully covered and the sparse branch — a
different solver, reached by graph size alone — had never been executed. The
new `test_sparse_path_matches_the_dense_answer` pins four graphs above the
threshold to the dense answer.

### 1.4 Not changed

- **`spectral_bipartition`'s `balance`** divides by `max(len(set1), len(set2))`
  and would raise on an empty side. Not reachable: the Fiedler vector is
  orthogonal to the constant vector (or to `D^(1/2)·1`), so its entries carry
  both signs unless it is identically zero.
- **`_kmeans` materialises an `(n, k, d)` distance array** each Lloyd
  iteration. Fine at the sizes here; the first thing to change if this is ever
  pointed at a corpus-scale graph.

---

## 2. The benchmark: 14 architectures

Eight architectures carry a planted ground truth; six are null controls where
the honest answer is "there is nothing to find". A method that scores well on
the planted partitions and *also* claims structure in an expander has not
earned the win, which is why the controls are there.

Baselines are all non-spectral, from networkx: Kernighan–Lin for cuts, greedy
modularity and label propagation for communities, and a random partition as
the floor.

### 2.1 Community recovery (ARI vs planted truth; 1.0 exact, 0.0 chance)

| architecture | n | k | k̂ | spec-N | spec-U | greedy | lpa | winner |
|---|---|---|---|---|---|---|---|---|
| sbm_2_strong | 180 | 2 | 2 | **1.000** | 1.000 | 1.000 | 1.000 | tie |
| sbm_4_medium | 200 | 4 | 4 | **1.000** | 1.000 | 0.987 | 1.000 | tie |
| sbm_2_weak | 200 | 2 | 2 | **0.638** | 0.000 | 0.237 | 0.000 | **spectral +0.40** |
| ring_of_cliques | 72 | 6 | 6 | **1.000** | 1.000 | 1.000 | 1.000 | tie |
| hierarchical_sbm | 160 | 4 | 2 | **1.000** | 0.983 | 0.489 | 0.710 | **spectral +0.29** |
| barbell | 140 | 2 | 10 | **1.000** | 1.000 | 0.733 | 0.750 | **spectral +0.25** |
| bipartite_corpus | 165 | 3 | 3 | 0.982 | 0.982 | **1.000** | 0.653 | greedy −0.02 |
| core_periphery | 160 | 2 | 3 | 0.001 | −0.034 | −0.006 | 0.000 | none (all at chance) |

**Spectral clustering never loses, and wins on three architectures.** On the
easy planted partitions everything ties at 1.000 — those cases prove
correctness and no advantage. The wins are concentrated where the structure is
hard:

- **`sbm_2_weak` (+0.40)** — near the detectability limit, spectral recovers
  0.638 where greedy modularity gets 0.237 and label propagation gets nothing.
- **`hierarchical_sbm` (+0.29)** — nested communities. Modularity's greedy
  merge commits to the wrong level of the hierarchy and cannot back out;
  the eigenvectors carry both levels at once.
- **`barbell` (+0.25)** — a single hard bottleneck, which is the shape Cheeger
  theory is *about*.

`core_periphery` is the negative result worth keeping: a dense core with a
sparse fringe looks like two groups to a human, and every method scores at
chance, correctly. It has no community structure — every periphery node hangs
off the core — and the spectrum agrees (§2.3 certifies it).

**The eigengap heuristic is the weak link.** It gets k right on 5 of 8, and
its failures are not near-misses: it says k=2 for the 4-community
`hierarchical_sbm` (defensible — there really are 2 super-communities) and
**k=10 for the barbell**, whose answer is 2. The bridge path contributes a
run of small eigenvalues that the heuristic reads as clusters. Do not wire k
selection to it without a sanity bound.

### 2.2 Normalized vs unnormalized — and why the obvious explanation is wrong

The largest single effect in the recovery table is `sbm_2_weak`: 0.638
normalized against **0.000** unnormalized. That reproduces across all 8 seeds
tested (normalized 0.53–0.77, unnormalized ≤0.001 every time), so it is not a
fluke.

The textbook explanation is degree heterogeneity — normalization exists to stop
hubs dominating. **The benchmark rules that out**: degCV is 0.20 for that graph,
essentially identical to `sbm_2_strong` (0.18) where both variants score 1.000.

The actual mechanism is **localization**, measured by the inverse participation
ratio of the Fiedler vector (`1/Σvᵢ⁴` for unit-norm v — the number of nodes
carrying the vector):

| architecture | degCV | support (norm) | support (unnorm) | spec-N | spec-U |
|---|---|---|---|---|---|
| sbm_2_strong | 0.18 | 170.4 | 176.1 | 1.000 | 1.000 |
| **sbm_2_weak** | 0.20 | 87.5 | **1.2** | **0.638** | **0.000** |
| barbell | 0.39 | 120.4 | 129.0 | 1.000 | 1.000 |
| core_periphery | 1.31 | 27.7 | 51.1 | 0.001 | −0.034 |

The unnormalized Fiedler vector collapses onto **1.2 of 200 nodes**. Its sign
carries no global bipartition because it has no global support. The normalized
one spreads over 87.5 and correlates 0.84 with the planted block (against 0.40
unnormalized); correlation with degree is 0.11 and 0.05 respectively, which
confirms degree is not what is happening.

**Practical rule:** default to `normalized=True`, and if you must use the
unnormalized Laplacian, check the Fiedler vector's support before trusting its
sign.

### 2.3 Cut quality and the Cheeger certificate

| architecture | kind | μ₂ | φ sweep | φ KL | φ rand | Cheeger bracket | hi/lo |
|---|---|---|---|---|---|---|---|
| sbm_2_strong | planted | 0.0644 | 0.0350 | 0.0350 | 0.5094 | [0.032, 0.359] | 11× |
| ring_of_cliques | planted | 0.0064 | 0.0050 | 0.0050 | 0.4963 | [0.003, 0.113] | 35× |
| barbell | planted | 0.0000 | 0.0003 | 0.0003 | 0.5158 | [0.000, 0.007] | 546× |
| **bipartite_corpus** | planted | 0.0265 | **0.0260** | 0.0933 | 0.5739 | [0.013, 0.230] | 17× |
| **core_periphery** | planted | 0.4699 | **0.3355** | 1.0000 | 0.5658 | [0.235, 0.969] | 4× |
| erdos_renyi | control | 0.4462 | 0.3210 | 0.3143 | 0.5090 | [0.223, 0.945] | 4× |
| random_regular | control | 0.2852 | 0.2472 | 0.2222 | 0.5296 | [0.143, 0.755] | 5× |
| **balanced_tree** | control | 0.0092 | **0.0127** | 0.0508 | 0.5726 | [0.005, 0.135] | 30× |

Across all 14 architectures the sweep cut and Kernighan–Lin each win 5 and tie
4 — a wash on the raw tally. The tally is the wrong summary, because the two
optimise different things:

**Kernighan–Lin is constrained to balanced halves and cannot express an
unbalanced cut at all.** Restricted to the architectures whose best cut is
unbalanced, the sweep cut wins. The two clearest cases are the ones that matter
most here:

- **`bipartite_corpus` (0.026 vs 0.093, 3.6× better)** — the document/entity
  shape of this project's own knowledge graph.
- **`balanced_tree` (0.013 vs 0.051, 4× better)** — snipping one subtree is
  the right cut and it is nowhere near balanced.
- **`core_periphery` (0.336 vs 1.000)** — KL's forced bisection cuts straight
  through the core and returns the worst possible answer.

Where the graph *does* split evenly, KL matches or narrowly beats the sweep cut
(`sbm_4_medium`: 0.095 vs 0.111). Timing does not separate them at n ≈ 180
(both single-digit ms); the sweep cut's advantage grows with size — measured
KL/sweep of 0.8× at n=200 rising to 2.0× at n=1600.

**The Cheeger lower bound is the underrated result.** μ₂/2 is a *proof of
absence*: no cut anywhere in the graph beats it. That is what distinguishes
"there is no bottleneck" from "I did not look hard enough", and no
local-search method can produce it at any price. On the controls:

```
erdos_renyi      μ₂=0.4462  certified: no cut below φ=0.223 exists
random_regular   μ₂=0.2852  certified: no cut below φ=0.143 exists
barabasi_albert  μ₂=0.3010  certified: no cut below φ=0.150 exists
watts_strogatz   μ₂=0.0146  a good cut is possible, and one was found at φ=0.0386
grid_2d          μ₂=0.0140  a good cut is possible, and one was found at φ=0.0440
balanced_tree    μ₂=0.0092  a good cut is possible, and one was found at φ=0.0127
```

The upper bound √(2μ₂) is much weaker — the bracket spans 4× to 546×, and it is
*widest* exactly where the cut is best (the barbell). As a certificate of
quality it is nearly useless; as a certificate of absence the lower bound is
exact and cheap. The bound held on all 14 architectures.

---

## 3. Verdict

**Reach for the spectrum when:**

1. **You need to know there is nothing to find.** The μ₂/2 lower bound is a
   proof of absence, and it is the one thing here no baseline can supply at
   any cost. This is the strongest result in the benchmark.
2. **The structure is weak, hierarchical, or nested.** +0.40 ARI at the
   detectability limit, +0.29 on nested communities — where greedy modularity's
   committed merges cannot recover.
3. **The natural cut is unbalanced.** 3.6× better conductance on the
   bipartite corpus shape, 4× on a tree. Balanced-bisection methods cannot
   express these cuts.
4. **You want one computation to answer several questions.** One eigendecomposition
   yields the component count, λ₂, the bipartition, the k-way clustering, the
   embedding, and the conductance certificate.

**Do not bother when:**

1. **The communities are obvious.** Everything ties at ARI 1.000 on strong
   planted partitions, and label propagation gets there in 2 ms against
   spectral's 6 ms.
2. **You need k and have no prior.** The eigengap heuristic missed on 3 of 8,
   including k̂=10 for a graph whose answer is 2.
3. **The graph is an expander.** Correct answer, no structure to report, and
   the shift-invert factorization is ~15× slower than the alternative there.

**For this project specifically:** `reports/spectral_applicability.md` proposes
five applications against `graphrag_server.py`. This benchmark supports A1
(connectivity health check — the component count and λ₂ are exact and cheap)
and A3 (bottleneck detection — the strongest measured advantage, on the
architecture that matches the knowledge graph). It weakens A2's implied
reliance on the eigengap to pick k: choose k another way, or bound it.

> **Resolved since.** All three are now implemented in `graphrag_server.py`
> (`connectivity()`, `topics()`, `bottleneck()`). A2's k problem is handled by
> gating the eigengap's answer on how decisively it won — measured across 18
> corpora with a planted topic count the heuristic was right every time at
> 5.1–23.4×, while a grid, a small-world ring, an expander and a single dense
> topic all landed at 1.0–1.8×, so a 3× threshold rejects the cases this
> benchmark caught it failing. Per-cluster conductance is carried alongside as
> an independent check. And it
adds a prerequisite none of them state — the knowledge graph is an
`nx.DiGraph`, every entry point here refuses one, and `to_undirected()` is
deliberately the caller's decision to make.
