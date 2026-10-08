# Benchmark evidence

This page is a curated summary of completed measurements. It is not a run log. Raw local experiment records, training harnesses, datasets and checkpoints are intentionally not distributed in this operator repository. Consequently these summaries are **not a self-contained reproduction bundle**. The source-result identifiers below identify the retained internal evidence; they are not public download links.

All tables distinguish the architecture, training regime and metric scope. Lower BPB is better; higher tokens/s is better. GiB = 2^30 bytes and MiB = 2^20 bytes. Results are exploratory, single-seed or finite timing screens, without confidence intervals.

## 1. Matched fresh 8K training

**Question:** What changes when some or all full-attention layers are replaced by TaoMemory under the same short training recipe?

All three runs completed on the same RTX 4090 with Linux, Python 3.12, PyTorch 2.9.1+cu128 and Triton 3.5.1. Training used BF16 and `torch.compile(dynamic=False)`, not manual inference CUDA Graphs.

| Recipe | Value |
|---|---|
| Backbone | 12 layers, width 768, 6 heads, head dimension 128, vocabulary 32,768 |
| Reference source | External nanochat revision `69ab2b93e9a209cc87771cae1e0ef2312301a96f`; not vendored or a package dependency |
| Data | Fixed ClimbMix revision `915333b4f8b8684f39aeaafea600fea6f43fb703`; training shards 00000–00031, validation shard 06542 |
| Initialization | Fresh seed 42; common named tensors and first input/target/cursor digests matched |
| Batch | 8,192 targets per microbatch; 64 accumulated microbatches; 524,288 targets/update |
| Training | 250 complete updates, 131,072,000 targets; complete 250-update schedule |
| Optimizer | Native MuonAdamW groups; unembedding LR .008, embedding .3, matrix .02, scalar .5, decay .28; common reference scaling |
| Schedule | 40 warmup updates, .65 warmdown fraction, .05 final LR fraction |
| Validation | 41,943,040 targets per pass, initial and final passes |
| CORE | Native centered aggregate over 22 tasks, maximum 500 examples/task; not accuracy |
| Hybrid | Full attention at 1-based layers 4, 8, 12; TaoMemory at the other nine |

| Metric | Native full attention | Pure TaoMemory | 3:1 Tao/full hybrid |
|---|---:|---:|---:|
| Final validation BPB ↓ | 1.107370515 | **1.100056009** | 1.104261593 |
| Final CORE ↑ | 0.047205268 | 0.053179418 | **0.056939284** |
| Parameters | 286,261,730 | 286,317,026 | 286,303,202 |
| All-update throughput, targets/s ↑ | **86,103.21** | 34,877.89 | 41,498.49 |
| Throughput excluding first update, targets/s ↑ | **88,063.41** | 40,770.35 | 47,786.95 |
| First update, s | 39.84 | 556.00 | 426.61 |
| Last 100 update median, s | 5.9534 | 12.8583 | 10.9671 |
| Sum of 250 update intervals, s | 1,522.27 | 3,758.03 | 3,158.48 |
| Training peak allocated, bytes | 6,573,232,128 | 8,226,425,344 | 7,774,356,992 |
| Training peak reserved, bytes | 6,629,097,472 | 9,588,178,944 | 8,854,175,744 |
| Whole-process wall including evaluation/save, s | 1,988.74 | 5,412.45 | 4,668.05 |

Pure TaoMemory's BPB is 0.66% lower than full attention; its compile-inclusive training throughput is 59.49% lower and training peak allocation is 25.15% higher. The hybrid reduces Tao's training cost but does not improve its BPB in this screen. The tiny parameter differences are disclosed rather than called exactly parameter-matched.

Allocator scope: the training rows use cumulative maxima recorded at update events, including initial validation before the first update. They are not peaks isolated to forward/backward/optimizer execution alone. Final evaluation is outside these update-event observations.

Limitations: sequential pure-Tao → hybrid → full-attention order; one seed; only 250 updates; common initial-input identity does not prove a full-stream hash comparison. This table says nothing about matched inference speed, full pretraining convergence or a general hybrid optimum. Full-attention inference and Tao's integrated Graph runtime were not compared in this experiment.

Evidence identifiers (SHA-256 of compact source-bound results):

- Full attention: `c00222a5e1b51b25c593567199873b3c06e74e6aac093fb1a70e829e209398a1`
- Pure Tao: `46c897fcae31c3abed89612597fd3f182c89c3e3336d70dbf6d8e0681ddc98f3`
- Hybrid: `59d9ca92ace7a0a8e22c4a0ae5b1e15392834db0b4969f75a9dae976efc2136d`

## 2. Common-target context-length screen

These are **different, historical 4K/250-update checkpoints**, initialized from a common pretrained model and trained under their historical schedule. They are not the fresh 8K models above. At each context length, four fixed held-out streams score the same final 1,024 targets per stream: 4,096 targets and 20,012 bytes total. We pool `total_nats / total_bytes / ln(2)`. Positions restart at zero and the context preceding the targets changes with length.

| Context | Raw exact KV BPB | Three-branch dual-softmax BPB | Tao coarse-index-only BPB |
|---|---:|---:|---:|
| 2K | **0.874061530** | 0.876883374 | 0.876530193 |
| 4K | **0.870593616** | 0.876825909 | 0.876641766 |
| 8K | 0.881927322 | 0.876614354 | **0.876398695** |
| 16K | 0.932686032 | 0.876689477 | **0.876374957** |
| 32K | 1.160957862 | 0.876870418 | **0.876566256** |
| 64K | 1.661353647 | 0.876767255 | **0.876415000** |
| 128K | 1.841504552 | 0.876692561 | **0.876543446** |

This demonstrates stable suffix scores for the compressed models on these four streams while the exact-KV model degrades beyond its trained context. It does **not** establish useful dependence on distant information, better long-context retrieval, or a comparison with a full-attention model trained for 128K. Repeated targets and lengths are correlated; the small index-only improvements are not seven independent confirmations. Exact KV is better at the trained short lengths.

Coarse-index-only result identifier: `969d03b801504e13f391e5111e66e720d5f9e8b242f01b9f5ff391251bff2425`. Its source-bound summary identifier: `3323e10c376523dd9dfd0f18c3c9180fe00f65f745cd905254ddee595c74535f`.

## 3. Retained cache, separately from process memory

Unique retained backing storage immediately after prefill, batch 1, 12 layers, width 768, BF16. These historical measurements precede the combined Graph integration. Raw and three-branch Tao are from one cost study; index-only is from a separate matched three-branch/index-only study with the same length/snapshot contract. Raw preallocated `L + 640` token capacity, including **22.50 MiB unused KV capacity at position L**. Values include retained routing/raw-K metadata and backing-storage overhead where applicable.

| Context L | Raw exact KV, MiB | Three-branch Tao, MiB | Coarse-index-only Tao, MiB |
|---|---:|---:|---:|
| 2K | 94.88 | 40.99 | 40.99 |
| 4K | 166.88 | 61.39 | 59.14 |
| 8K | 310.88 | 102.21 | 95.46 |
| 16K | 598.88 | 183.85 | 168.10 |
| 32K | 1,174.88 | 347.12 | 313.37 |
| 64K | 2,326.88 | 673.65 | 603.90 |
| 128K | 4,630.88 | 1,326.72 | 1,184.97 |

This is retained cache storage, **not training memory, total inference VRAM or allocator reservation**. Fine matrices remain resident as context grows. Model weights, temporary tensors and Graph execution pools require additional memory. Do not interpret a smaller cache as an automatic throughput win. Index-only cost result identifier: `4de2832eba20cef089acabc7c97a9ae638a0a9be2349336ed34231d6fc894ca9`.

## 4. Integrated page memo + regional CUDA Graph

The same source, weights and inputs were compared with Graph disabled/enabled. Both arms use page-normalization memoization. Eight isolated ABBA children cover 16K and 128K; each performs a full warm pass and a fresh clean prompt followed by 640 real advancing tokens. Capture, copies, waits, route refresh, sealing and teardown are inside the timing. Runtime: RTX 4090, BF16, batch 1, head dimension 128.

| Prompt | Memo decode tok/s | Memo+Graph decode tok/s | Change | Memo prefill tok/s | Memo+Graph prefill tok/s |
|---|---:|---:|---:|---:|---:|
| 16K | 113.330 | 123.129 | +8.65% | 3,932.156 | 3,798.211 |
| 128K | 107.192 | 123.816 | +15.51% | 3,425.309 | 3,518.180 |

Both neighboring decode pairs improved at both lengths. Prefill code is unchanged; its mixed changes are observations, not evidence of a prefill optimization. Two clean samples per arm/length are insufficient for broad statistical claims. This is **regional history graph execution**, not whole-model graph capture or `torch.compile`, and not a matched exact-KV speed comparison.

| Memory at L+640 / phase | 16K memo | 16K memo+Graph | 128K memo | 128K memo+Graph |
|---|---:|---:|---:|---:|
| Unique semantic cache after charged teardown, bytes | 187,723,392 | 187,723,392 | 1,253,992,128 | 1,253,992,128 |
| Decode peak allocated, bytes | 1,099,070,976 | 1,100,558,336 | 2,222,315,008 | 2,222,180,352 |

All 1,280 paired warm-query full-logit hashes and recorded clean boundary fingerprints matched. All 1,056 lifecycle forwards completed, with observed solo/interleaved and reset/import agreement and no cross-request aliases. These are tested trajectories, not universal equivalence proofs. The lifecycle observation exposed 413,568 bytes of memo/staging/output buffers; graph-private scratch was not separately attributed and is included in process peaks.

Original result archive identifier: `0228991b461655d884b949689e6ac56377ba7365b842ed1e63b07adb3a94d029`. Tested source bundle identifier: `a198bad772ae2165322f0de001cd521fcef56a67d419297f9826e69385d106fd`. The public export also incorporates the subsequent verified singleton-visibility training repair; export packaging is not a new performance measurement.

## What remains unsolved

Natural-text BPB does not measure precise key/value retrieval. In a separate 4K task-trained assay, exact KV answered 48/48 sealed-history records correctly, while the original three-branch dual-softmax model answered 24/48. Both answered 48/48 records inside the exact local window. This is a task-specific historical comparison, not a new score for every configuration in this release. The historical record-value positions were 527–663 with the final query at 4095; local record values were 3727–3863. The difference is therefore actual memory placement, not an unspecified notion of “near” or “far.”

System-prompt retention, broad instruction following, robust distant binding, multi-seed convergence and optimized exact-KV inference comparisons remain open. This repository releases a usable research operator and evidence of a cache/quality tradeoff; it does not claim that compressed memory has solved long-context understanding.
