"""Fused candidate products with the original Torch channel reduction order."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scores(HV, W, IDS, PAGES, VISIBLE, OUT,
            Q: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
            S: tl.constexpr, U: tl.constexpr, D: tl.constexpr,
            STATIC_VISIBLE: tl.constexpr):
    row = tl.program_id(0)
    u = tl.program_id(1) * 4 + tl.arange(0, 4)
    d = tl.arange(0, D)
    h = row % H
    b = row // (Q * H)
    fine = tl.load(IDS + row * U + u, u < U, other=0)
    page = tl.load(PAGES + row * U + u, u < U, other=0)
    if STATIC_VISIBLE >= 0:
        visible = STATIC_VISIBLE
    else:
        visible = tl.load(VISIBLE + b)
    valid = (u < U) & (fine < visible)
    hv = tl.load(HV + (row * P + page[:, None]) * D + d[None, :],
                 valid[:, None], other=0).to(tl.float32)
    w = tl.load(W + ((b * H + h) * S + fine[:, None]) * D + d[None, :],
                valid[:, None], other=0).to(tl.float32)
    channel = (hv * w).to(tl.bfloat16)
    tl.store(OUT + (row * U + u[:, None]) * D + d[None, :], channel,
             (u < U)[:, None])


def candidate_scores(hv, weight, ids, pages, visible):
    b, q, h, p, d = hv.shape
    u = ids.shape[-1]
    static_visible = visible if isinstance(visible, int) else -1
    visible_tensor = None if isinstance(visible, int) else visible.contiguous()
    channel = torch.empty((b, q, h, u, d), dtype=hv.dtype, device=hv.device)
    _scores[(b * q * h, triton.cdiv(u, 4))](
        hv.contiguous(), weight.contiguous(), ids.contiguous(),
        pages.contiguous(), visible_tensor, channel,
        q, h, p, weight.shape[2], u, d, static_visible)
    # The r4 tl.sum tree changed BF16 boundary scores. Preserve Torch's
    # original reduction on the same contiguous BF16 products instead.
    scores = channel.sum(-1)
    limit = visible if isinstance(visible, int) else visible[:, None, None, None]
    return scores.masked_fill(ids >= limit, -torch.inf)
