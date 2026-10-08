# Configuration

```python
from tao_attention import TaoConfig
from tao_attention.presets import index_only_page_memo

config = index_only_page_memo()             # eager history, memo enabled
captured = index_only_page_memo(graph=True) # same arithmetic, regional capture
restored = TaoConfig(**config.export())
```

Use the named preset explicitly. Historical `TaoConfig()` defaults retain a
coarse value branch and are deliberately not changed by this export.

| Preset field | Value | Meaning |
|---|---:|---|
| `fine_tokens` | 256 | Completed fine-page span |
| `local_keep` | 512 | Exact local retention after a seal |
| `coarse_group` | 8 | Fine pages per coarse routing summary |
| `route_period` | 24 | Base staggered refresh schedule |
| `layer_stagger` | 2 | Layer-dependent refresh phase |
| `budget_root_scale` | 2 | Selected-page budget scale |
| `budget_offset` | 2 | Budget offset |
| `fine_writer` | `softmax` | Token-axis softmax compressed writer |
| `coarse_index_only` | `True` | Coarse z routes; no coarse value bank/read |
| `q1_page_normalization_memo` | `True` | Query-independent normalization reuse |
| `q1_history_region_graph` | default `False` | Optional q1 regional execution |

For positive page count n, the budget is min(n, ceil(sqrt(2n)) + 2).
Values remain independent across heads. Fine matrices, raw descriptors, local
SDPA and fusion preserve their original rounding/reduction order. The preset
adds no learned parameters; the caller owns fusion parameters and projections.

Compatibility writer implementations and internal diagnostic fields remain in
the source to avoid altering the established implementation. They are not part
of the recommended/tested public preset. Do not enable them based on benchmark
results for this preset. Arbitrary geometry changes are not a validated envelope.
No system/role mask, prefix pinning, attention sink or instruction-retention policy
is implemented. Compressed and selected history can omit useful information.
