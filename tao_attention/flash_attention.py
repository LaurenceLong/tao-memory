"""Causal SDPA adapter for Tao's [batch, tokens, heads, width] tensors.

Only the causal full-local operator is needed by this package. The caller owns
persistent KV; this module neither imports an external kernel registry nor
changes the registered attention scale or causal mask.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

USE_FA3 = False


def flash_attn_func(q, k, v, causal=True, window_size=(-1, 0)):
    if not causal or window_size != (-1, 0):
        raise ValueError("standalone Tao supports causal full-local attention only")
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    if q.shape[1] == k.shape[1]:
        output = F.scaled_dot_product_attention(query, key, value,
                                                is_causal=True)
    elif q.shape[1] == 1:
        # One new query is the final position and may read every supplied
        # exact key. Use the same direct noncausal SDPA call as the reference.
        output = F.scaled_dot_product_attention(query, key, value,
                                                is_causal=False)
    else:
        # Query rows represent the suffix of the concatenated old/new keys.
        rows = k.shape[1] - q.shape[1] + torch.arange(q.shape[1], device=q.device)
        columns = torch.arange(k.shape[1], device=q.device)
        allowed = columns[None, :] <= rows[:, None]
        output = F.scaled_dot_product_attention(query, key, value,
                                                attn_mask=allowed, is_causal=False)
    return output.transpose(1, 2)


flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)
