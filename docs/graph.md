# Optional regional CUDA Graph execution

`index_only_page_memo(graph=True)` wraps the same memo-backed q1 history callable
used by eager execution. It does not capture the model, local SDPA, route refresh,
fusion, prefill or training. No separate Graph numerical formula is maintained.
Fresh raw queries and selected IDs are staged into each request/layer's buffers.
Three eligible real queries execute eagerly on its side stream; the next real
query captures and replays once. These queries serve the caller; there is no
hidden disposable warmup. Misses/capture/synchronization are real runtime costs.

Use serial inference-mode calls on one consistent caller CUDA stream/device per
request. Do not nest capture, concurrently mutate a request, share states or run
unrelated concurrent CUDA work during capture. Independent requests may be
interleaved serially. `state.release_execution()` is required when abandoning a
live request; `reset()` also clears semantic state. Exceptions propagate without
an automatic eager fallback.

One epoch graph is retained per layer. Seal/shape/memo-address changes invalidate
old execution references before semantic storage replacement. Refresh supplies
fresh IDs. Export contains no graphs or captured pointers; import/reset starts
with empty execution resources. Page memo is prepared outside capture with its
original Torch operations and participates in captured-address identity.

Graph staging, outputs and private pools are extra execution memory beyond the
semantic cache. Allocator reservations can survive release. Visible tensor bytes
alone are not total pool cost; account for live snapshots and process peaks.
Measured Graph coverage is Linux/4090/BF16, batch1/H6/D128,16K and128K with actual
advancing q1 suffixes. Wider shape/device/concurrency coverage is not claimed.
See [benchmarks](benchmarks.md) for curated results and comparison scope. Source
packaging/static CI is not a fresh runtime validation of an installed release.
