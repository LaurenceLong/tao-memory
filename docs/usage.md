# Functional API

All five inputs `q`, `k`, `v`, `raw_q`, `raw_k` use `[batch, tokens, heads, width]`.
The documented streaming envelope is batch1, BF16 CUDA, equal Q/K/V heads and
width64 or128. `raw_q/raw_k` are caller projections before its normalization and
positional transform; `q/k` are the transformed exact-local operands. `v` is the
caller's final effective value. Output has the same layout, before output
projection. Fusion a/b have shape `[heads, width, 3]`; alpha is a Python history
scale. The core supplies no projections, positions, embeddings, loss or optimizer.

```python
from tao_attention import tao_attention, new_state, tao_attention_prefill
from tao_attention import tao_attention_decode, StreamCache
from tao_attention.presets import index_only_page_memo
config = index_only_page_memo()

# Differentiable full-sequence training; caller backpropagates its scalar loss.
heads = tao_attention(q, k, v, raw_q=raw_q, raw_k=raw_k,
    fusion_a=a, fusion_b=b, alpha=alpha, config=config)

# Inference: one state per request AND layer, under torch.inference_mode().
state = new_state(layer_index=0, config=config)
heads, state = tao_attention_prefill(q_new, k_new, v_new,
    raw_q=raw_q_new, raw_k=raw_k_new, fusion_a=a, fusion_b=b,
    alpha=alpha, state=state, position=state.get_pos())
heads, state = tao_attention_decode(q_one, k_one, v_one,
    raw_q=raw_q_one, raw_k=raw_k_one, fusion_a=a, fusion_b=b,
    alpha=alpha, state=state, position=state.get_pos())
```

Only supply NEW tokens. Decode requires exactly one token. Use the same absolute
position for caller positional transforms and the state cursor. Prefill splits
at causal serve/seal boundaries: the first256 tokens seal after the first768
positions have been served. Calls mutate and advance state; do not also advance
it yourself or refeed the old prefix. There is no padding mask. Never share state
between concurrent requests or layers.

Training uses fresh routing for each query; incremental inference uses staggered
refresh. Training and cached inference are not promised bitwise equivalent.
Use the stateless training API, not inference cache mutation, for gradients.
The singleton training visibility fix avoids scalar device rematerialization in
checkpointed compiled backward; multi-group visibility remains tensor-valued.

```python
snapshot = state.export()
independent = StreamCache.from_export(snapshot)
state.release_execution() # retire execution resources; semantic cursor survives
state.reset()             # discard semantic state and return to position zero
```

Snapshots clone semantic tensors, consuming additional memory. Memo/Graph
execution objects are not serialized. Import rebuilds from real subsequent
queries, keeps original device placement and creates independent state. Save
model weights/config separately. Do not mutate internal C/z banks or use an
untrusted snapshot. Optional `local_flash` must implement the core's causal
suffix-alignment contract; the included SDPA callback is the supported default.

[Generic examples](../examples/) demonstrate integration without any backbone.
A nanoGPT/NanoChat or other adapter should own its projections/position transform,
pass tensors into this API and bind one state per layer. No backbone or data
loader is shipped; training data/cursor recovery belongs to the caller.
