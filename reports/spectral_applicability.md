# Spectral Applicability: From `spectral_graphing.md` to This Project

This report maps the verified findings of `reports/spectral_graphing.md` onto the
concrete components of the langgraph-agent project. Every application below cites
the finding it builds on, names the component it touches, states the spectral
technique, and gives the expected benefit. No production code is changed by this
report; it is a proposal document only.

The knowledge graph on disk (`knowledge/knowledge_graph.json`) is a networkx
node-link JSON: a directed graph (`"directed": true`) whose nodes carry
`"type": "document"` or `"type": "entity"`, and whose edges all carry
`"relation": "mentions"` running document -> entity. It is loaded in
`graphrag_server.py` by `nx.readwrite.json_graph.node_link_graph` into an
`nx.DiGraph`. This bipartite document/entity shape is the object every spectral
technique below would operate on.

## A1. Connectivity health check on the knowledge graph

- **Finding cited:** Section 1 (Laplacian Eigenstructure) of
  `reports/spectral_graphing.md`: the number of zero eigenvalues of `L = D - A`
  equals the number of connected components, and the graph is connected iff the
  Fiedler value `lambda_2 > 0`. Verified on the karate club (`lambda_2 = 1.1871`).
- **Component touched:** `graphrag_server.py` -- the `stats()` method, which
  currently reports only node/edge/document/chunk counts.
- **Spectral technique:** Compute the Laplacian spectrum of the undirected
  projection of the knowledge graph (`self.graph.to_undirected()`), count the
  near-zero eigenvalues, and report `lambda_2`.
- **Expected benefit:** `stats()` gains a structural health signal. A reindex
  that silently drops edges (e.g. an entity-extraction regression in
  `add_document`) shows up as a rising component count or a collapsing
  `lambda_2` long before it shows up in search quality. The count of zero
  eigenvalues is an exact component count, so it also detects orphaned
  documents that no entity links back to the corpus.

## A2. Community structure over documents and entities

- **Finding cited:** Section 4 (Spectral Clustering): `k` weakly-coupled
  clusters produce `k` near-zero eigenvalues of the normalized Laplacian before
  an eigengap, and k-means on the row-normalized spectral embedding recovers
  the communities (36/36 nodes recovered on a 3-cluster planted partition).
- **Component touched:** `graphrag_server.py` -- a new read-only method
  alongside `neighborhood()` / `query_graph()`, and the data in
  `knowledge/knowledge_graph.json` it reads.
- **Spectral technique:** Normalized-Laplacian spectral embedding
  (Ng-Jordan-Weiss) + k-means, with `k` chosen from the eigengap.
- **Expected benefit:** The console currently shows one node's neighbourhood at
  a time. Spectral clustering would group documents and their entities into
  topic communities (e.g. all agent-runtime files vs. all GraphRAG files),
  giving a whole-corpus map that degree-filtered neighbourhood sweeps cannot.
  Because the graph is bipartite, clusters naturally mix documents with the
  entities that define them, which is exactly the structure a reader wants.

## A3. Bottleneck / bridge detection between topic areas

- **Finding cited:** Section 3 (Cheeger's Inequality): the sweep cut over the
  normalized Fiedler vector has conductance within `[mu_2/2, sqrt(2*mu_2)]`,
  and a small `mu_2` certifies a genuine bottleneck (sweep cut recovered the
  planted cut at k = 15 with conductance 0.0357).
- **Component touched:** `graphrag_server.py` -- read-only analysis over
  `self.graph`; results could surface in `stats()` or a new diagnostic method.
- **Spectral technique:** Normalized Fiedler vector + sweep cut for minimum
  conductance.
- **Expected benefit:** Identifies the narrow bridges of the knowledge graph --
  the few entities (or documents) through which two otherwise separate topic
  areas connect. Those bridge nodes are high-value: they are the terms a search
  should expand on when a query straddles two areas, and the documents whose
  deletion would fragment the corpus. Degree alone does not find them; the
  sweep cut does, with a guarantee.

## A4. Fiedler sign cut as a cheap two-way split for the console

- **Finding cited:** Section 2 (Fiedler Partitioning): the sign of the Fiedler
  vector is an approximate minimum-ratio bipartition (15/15 planted nodes
  recovered, `lambda_2 = 0.4355`).
- **Component touched:** `graphrag_server.py` -- the `neighborhood()` method,
  which today prunes only by `min_degree`.
- **Spectral technique:** Fiedler-vector sign bipartition on the swept
  subgraph.
- **Expected benefit:** When a neighbourhood sweep returns a large subgraph,
  the sign cut gives an immediate, principled two-way split for display --
  "these nodes cluster with the centre, these are peripheral" -- without any
  clustering library. It is the cheapest spectral technique in the report (one
  eigenvector) and so suits an interactive console path.

## A5. Spectral deduplication / near-duplicate entity detection

- **Finding cited:** Section 0 (Interpretation, finite-model-theory sense) and
  Section 1: two nodes with identical neighbourhoods are indistinguishable to
  any spectral method (they are cospectral twins -- the `C^2` / Weisfeiler-Lehman
  limit), and the Laplacian spectrum is invariant under relabelling of such
  twins.
- **Component touched:** `knowledge/knowledge_graph.json` -- the entity nodes
  minted by `add_document` in `graphrag_server.py`.
- **Spectral technique:** Compare rows of the spectral embedding (Section 4);
  entities with near-identical embedding rows have near-identical structural
  roles.
- **Expected benefit:** `add_document` mints an entity for every capitalised
  token, so near-duplicate entities (e.g. "Builder" vs. "Builders") accumulate.
  Embedding-row proximity flags candidates for merging. The finite-model-theory
  finding is the honest caveat: exact structural twins cannot be told apart by
  the spectrum alone, so this detects *near*-duplicates, not exact ones -- which
  is the useful case anyway.

## A6. What does not transfer

- **Finding cited:** Section 0 (finite-model-theory sense): cospectral graphs
  agree on all `C^2` sentences, so no purely spectral feature distinguishes
  them.
- **Component touched:** `graph.py` and `state.py` -- the agent routing graph
  and its state schema.
- **Spectral technique:** none applicable.
- **Expected benefit:** This is a scoping result, not an application. The agent
  graph in `graph.py` is a small fixed control-flow DAG (4 nodes), and
  `state.py` is a typed dictionary schema; neither has the size or the
  community structure that spectral methods exploit, and any spectral signature
  of the routing graph would be blind to the semantic differences between
  agents (cospectral routing topologies are interchangeable spectrally). The
  spectral findings apply to the *knowledge* graph, not the *agent* graph, and
  this report deliberately proposes nothing for `graph.py` or `state.py`.

## Summary

| # | Finding (spectral_graphing.md) | Component | Technique | Benefit |
|---|-------------------------------|-----------|-----------|---------|
| A1 | Sec. 1 Laplacian eigenstructure | `graphrag_server.py` `stats()` | zero-eigenvalue count, `lambda_2` | structural health check on reindex |
| A2 | Sec. 4 spectral clustering | `graphrag_server.py` + `knowledge/knowledge_graph.json` | normalized embedding + k-means | whole-corpus topic communities |
| A3 | Sec. 3 Cheeger's inequality | `graphrag_server.py` | Fiedler sweep cut | bridge/bottleneck detection |
| A4 | Sec. 2 Fiedler partitioning | `graphrag_server.py` `neighborhood()` | Fiedler sign cut | cheap two-way split for display |
| A5 | Sec. 0 + Sec. 1 (cospectrality) | `knowledge/knowledge_graph.json` | embedding-row proximity | near-duplicate entity detection |
| A6 | Sec. 0 (C^2 limit) | `graph.py`, `state.py` | none | scoping: agent graph is not a spectral target |

All techniques use only numpy/scipy/networkx, the same stack verified in
`reports/spectral_graphing.md`, and all operate on the undirected projection of
the existing `nx.DiGraph`, so none requires a schema change to
`knowledge/knowledge_graph.json`.
