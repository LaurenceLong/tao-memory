"""Functional Tao attention. The caller owns all learned projections and gates.

Inputs have shape [batch, tokens, heads, head_width]. ``raw_q`` and ``raw_k``
are the projections before the caller's Q/K transform; ``q`` and ``k`` are the
local-attention tensors after it. The supported streaming request is batch 1,
contiguous causal tokens, BF16, head width 64 or 128. There is no padding mask.
"""

import torch
import torch.nn.functional as F

from .flash_attention import flash_attn
from .fusion_pointwise import fusion_scores, fusion_weighted_sum
from .tao_stream import StreamCache, read_training
from .config import TaoConfig


DEFAULT_CONFIG = TaoConfig()


def _fuse_training(local, raw_q, coarse, fine, coarse_present, fine_present,
                   fusion_a, fusion_b, alpha):
    dtype = raw_q.dtype
    score = (fusion_a.to(dtype) * raw_q.unsqueeze(-1)).to(dtype)
    score = (score + fusion_b.to(dtype)).to(dtype)
    history_count = coarse_present.to(torch.int32) + fine_present.to(torch.int32)
    local_bias = history_count.clamp_min(1).to(dtype).log().to(dtype)
    score = torch.stack(((score[..., 0] + local_bias).to(dtype),
                         score[..., 1], score[..., 2]), dim=-1)
    score = torch.stack((score[..., 0],
                         score[..., 1].masked_fill(~coarse_present, -torch.inf),
                         score[..., 2].masked_fill(~fine_present, -torch.inf)), dim=-1)
    rho = F.softmax(score, dim=-1).to(dtype)
    gain = torch.full((), 2 * alpha, dtype=dtype, device=raw_q.device)
    local_term = (rho[..., 0] * (2 * local).to(dtype)).to(dtype)
    coarse_term = (rho[..., 1] * (gain * coarse).to(dtype)).to(dtype)
    fine_term = (rho[..., 2] * (gain * fine).to(dtype)).to(dtype)
    fused = ((local_term + coarse_term).to(dtype) + fine_term).to(dtype)
    return torch.where(coarse_present | fine_present, fused, local)


def tao_attention(q, k, v, *, raw_q, raw_k, fusion_a, fusion_b, alpha,
                  config=DEFAULT_CONFIG, local_flash=None):
    """Full differentiable sequence read with fresh routing at every query."""
    local, coarse, fine, _ = read_training(
        raw_q, raw_k, v, q, k,
        flash_attn if local_flash is None else local_flash, route_interval=1,
        config=config)
    tokens = q.shape[1]
    positions = torch.arange(tokens, device=q.device)[None, :, None, None]
    return _fuse_training(local, raw_q, coarse, fine,
                          (positions >= config.first_coarse) & (not config.coarse_index_only),
                          positions >= config.first_fine,
                          fusion_a, fusion_b, alpha)


def new_state(layer_index, config=DEFAULT_CONFIG):
    """Independent mutable per-request, per-layer inference state."""
    return StreamCache(layer_index, route_interval=2, fusion_mode="pointwise",
                       fused_fine=True, config=config)


def _history_path_boundary(state, coarse, fine, coarse_on, fine_on):
    """Diagnostic only: keep all history computation, then use absent-history fusion."""
    if state.history_path_ablation:
        return torch.zeros_like(coarse), torch.zeros_like(fine), False, False
    return coarse, fine, coarse_on, fine_on


def tao_attention_prefill(q, k, v, *, raw_q, raw_k, fusion_a, fusion_b,
                          alpha, state, position, local_flash=None):
    """Advance only supplied new tokens; read before each actual 768 seal.

    The state owns its tensors and is updated in place, then returned. A caller
    must not share one state between requests or call layers out of order.
    """
    if position != state.position:
        raise ValueError("position must match this request's causal cursor")
    tokens = q.shape[1]
    if tokens and tokens == state.next_chunk(tokens):
        local, coarse, fine, coarse_on, fine_on = state.process(
            0, raw_q, raw_k, q, k, v, local_flash=local_flash,
            allow_page_memo=tokens == 1)
        coarse, fine, coarse_on, fine_on = _history_path_boundary(
            state, coarse, fine, coarse_on, fine_on)
        if coarse_on or fine_on:
            score = fusion_scores(raw_q, fusion_a, fusion_b, coarse_on, fine_on)
            rho = F.softmax(score, dim=-1).to(q.dtype)
            gain = torch.full((), 2 * alpha, dtype=q.dtype, device=q.device)
            # Keep ordinary presence/gates and every cast in the existing final sum.
            if state.history_value_ablation:
                coarse, fine = torch.zeros_like(coarse), torch.zeros_like(fine)
            output = fusion_weighted_sum(rho, local, coarse, fine, gain)
            state.fusion_pointwise_calls += 1
        else:
            output = local
            state.fusion_no_history_local_returns += 1
        state.advance(tokens)
        return output, state
    outputs = []
    start = 0
    while start < q.shape[1]:
        width = state.next_chunk(q.shape[1] - start)
        end = start + width
        local, coarse, fine, coarse_on, fine_on = state.process(
            0, raw_q[:, start:end], raw_k[:, start:end],
            q[:, start:end], k[:, start:end], v[:, start:end],
            local_flash=local_flash)
        coarse, fine, coarse_on, fine_on = _history_path_boundary(
            state, coarse, fine, coarse_on, fine_on)
        if coarse_on or fine_on:
            score = fusion_scores(raw_q[:, start:end], fusion_a, fusion_b,
                                  coarse_on, fine_on)
            rho = F.softmax(score, dim=-1).to(q.dtype)
            gain = torch.full((), 2 * alpha, dtype=q.dtype, device=q.device)
            # Keep ordinary presence/gates and every cast in the existing final sum.
            if state.history_value_ablation:
                coarse, fine = torch.zeros_like(coarse), torch.zeros_like(fine)
            output = fusion_weighted_sum(rho, local, coarse, fine, gain)
            state.fusion_pointwise_calls += 1
        else:
            output = local
            state.fusion_no_history_local_returns += 1
        outputs.append(output)
        state.advance(width)
        start = end
    return torch.cat(outputs, dim=1), state


def tao_attention_decode(q, k, v, *, raw_q, raw_k, fusion_a, fusion_b,
                         alpha, state, position, local_flash=None):
    """One-token version of prefill; no old token projection is recomputed."""
    if q.shape[1] != 1:
        raise ValueError("decode accepts exactly one new query/key/value")
    return tao_attention_prefill(q, k, v, raw_q=raw_q, raw_k=raw_k,
                                 fusion_a=fusion_a, fusion_b=fusion_b,
                                 alpha=alpha, state=state, position=position,
                                 local_flash=local_flash)
