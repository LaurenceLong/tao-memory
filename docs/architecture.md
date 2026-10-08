# Architecture and integration boundary

TaoMemory combines exact causal local attention with compressed, selected
history. It returns head outputs to a caller-owned output projection; it is not a
Transformer backbone or a pretrained model.

For each completed256-token fine page, the default writer forms a dense per-head
matrix from token-axis softmax of raw keys and effective values. A per-channel
log-partition descriptor accompanies it. Fine pages remain resident even when a
query does not select them. Coarse summaries group eight fine descriptors for
hierarchical routing; the recommended index-only preset stores no coarse value
matrix and contributes no coarse value output. Values remain independent by head.

Exact local attention retains512 tokens after sealing and serves a full768-token
local span before retiring its oldest256. History selection uses query-dependent
scores and a sqrt(2n)+2 budget capped by eligible page count. Anchor/visibility
rules and staggered streaming refresh are part of the implementation. Fusion
combines available local/history outputs using caller-owned a/b parameters and
history scale. Index-only coarse absence affects this presence-aware fusion.

The differentiable training path uses checkpointed groups and fresh query routes.
The inference path mutates a per-request, per-layer cache and only consumes newly
arrived tokens. These paths have different route-refresh schedules: do not claim
full-sequence training and streaming are bitwise equivalent.

Page memo removes repeated query-independent normalization on eligible q1 reads.
Optional regional CUDA Graph executes the same callable with fresh inputs; it
adds execution workspace, not semantic memory capacity. Seals/reset/import obey
explicit ownership and invalidation. See [usage](usage.md) and [Graph](graph.md).

No role masks, system-prefix pinning, exact long-range token retention, automatic
checkpoint conversion, tokenizer or loader is supplied. Compression and routing
can lose information. Benchmarks describe measured configurations rather than a
guarantee of instruction following, arbitrary geometry or convergence.
