"""Grouped selected-only BF16 tensor-core reads with shared causal fine banks.

Forward and dA reuse each C tile across 32 selected queries. The original
per-query inference kernel remains below. Routing and all cast locations around
weights/normalization remain in tao_stream.py. No query-expanded C is formed.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_forward(A, I, C, Y, Q: tl.constexpr, H: tl.constexpr,
                      M: tl.constexpr, S: tl.constexpr,
                      D: tl.constexpr, E_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    e = tl.program_id(1) * E_BLOCK + tl.arange(0, E_BLOCK)
    d = tl.arange(0, D)
    h = row % H
    q = (row // H) % Q
    b = row // (H * Q)
    selected_base = ((b * Q + q) * H + h) * M
    acc = tl.full((E_BLOCK,), 0, tl.float32)
    for m in range(M):
        segment = tl.load(I + selected_base + m)
        a = tl.load(A + (selected_base + m) * D + d).to(tl.float32)
        c_offset = (((b * H + h) * S + segment) * D + d[:, None]) * D + e[None, :]
        c = tl.load(C + c_offset, e[None, :] < D, other=0).to(tl.float32)
        acc += tl.sum(a[:, None] * c, axis=0)
    tl.store(Y + row * D + e, acc, e < D)


@triton.jit
def _selected_bounds(SORTED, START, END, N: tl.constexpr, S: tl.constexpr,
                     TOTAL: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    # Every lane keeps the same vector shape throughout both binary searches.
    # Avoid Inductor's generic bucketize helper and its loop-carried broadcast.
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = slot < TOTAL
    row, key = slot // S, slot % S
    lower_lo = tl.full((BLOCK,), 0, tl.int32)
    lower_hi = tl.full((BLOCK,), N, tl.int32)
    upper_lo = tl.full((BLOCK,), 0, tl.int32)
    upper_hi = tl.full((BLOCK,), N, tl.int32)
    for _ in range(STEPS):
        lower_mid = (lower_lo + lower_hi) // 2
        upper_mid = (upper_lo + upper_hi) // 2
        lv = tl.load(SORTED + row * N + lower_mid,
                     valid & (lower_mid < N), other=2147483647)
        uv = tl.load(SORTED + row * N + upper_mid,
                     valid & (upper_mid < N), other=2147483647)
        below = lv < key
        at_or_below = uv <= key
        lower_lo = tl.where(below, lower_mid + 1, lower_lo)
        lower_hi = tl.where(below, lower_hi, lower_mid)
        upper_lo = tl.where(at_or_below, upper_mid + 1, upper_lo)
        upper_hi = tl.where(at_or_below, upper_hi, upper_mid)
    # Public bound tensors retain the original int64 dtype and contiguous
    # [batch, head, fine] layout consumed by all three contraction kernels.
    tl.store(START + slot, lower_lo, valid)
    tl.store(END + slot, upper_lo, valid)


class _SelectedContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, channel, selected_id, fine_c):
        b, q, h, m, d = channel.shape
        # The grouped forward dot is [32,D] x [D,32]; grad-A has the same
        # reduction width, and grad-C is tiled [32,32] x [32,32]. D=64 and
        # D=128 both satisfy the tensor-core block dimensions. All grids below
        # derive from D, so this is an actual D64 forward/backward kernel path.
        if (d not in (64, 128) or fine_c.shape[1] != h or b % fine_c.shape[0] != 0
                or fine_c.shape[3:] != (d, d)
                or selected_id.shape != (b, q, h, m)
                or channel.dtype != torch.bfloat16
                or fine_c.dtype != torch.bfloat16):
            raise ValueError("selected fine contraction requires BF16 D64 or D128 tensors")
        channel = channel.contiguous()
        selected_id = selected_id.contiguous()
        fine_c = fine_c.contiguous()
        s = fine_c.shape[2]
        bank_b = fine_c.shape[0]
        # At M=1 reshape can preserve strides (H*Q,1,H). Sort preserves
        # dense input strides, while the kernels below use contiguous offsets.
        by_head = selected_id.permute(0, 2, 1, 3).reshape(b, h, q * m).contiguous()
        sorted_ids, order = torch.sort(by_head, dim=-1, stable=True)
        starts = torch.empty((b, h, s), device=selected_id.device, dtype=torch.int64)
        ends = torch.empty_like(starts)
        _selected_bounds[(triton.cdiv(b * h * s, 128),)](
            sorted_ids.contiguous(), starts, ends, q * m, s, b * h * s, (q * m).bit_length(), 128)
        pairs = torch.empty((b, q, h, m, d), device=channel.device, dtype=torch.float32)
        output = torch.empty((b, q, h, d), device=channel.device, dtype=channel.dtype)
        _grouped_forward[(b * h * s, triton.cdiv(d, 32))](
            channel, fine_c, order, starts, ends, pairs, q, h, m, s, d, bank_b, 32)
        _sum_selected[(b * q * h,)](pairs, output, m, d)
        ctx.save_for_backward(channel, selected_id, fine_c, order, starts, ends)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        channel, selected_id, fine_c, order, starts, ends = ctx.saved_tensors
        b, q, h, m, d = channel.shape
        s = fine_c.shape[2]
        bank_b = fine_c.shape[0]
        grad_output = grad_output.contiguous()
        grad_channel = grad_fine_c = None
        if ctx.needs_input_grad[0]:
            grad_channel = torch.empty_like(channel)
            _grouped_grad_a[(b * h * s, triton.cdiv(d, 32))](
                fine_c, grad_output, order, starts, ends, grad_channel,
                q, h, m, s, d, bank_b, 32)
        if ctx.needs_input_grad[2]:
            # Sort only the selected IDs.  Every key in ORDER is one selected
            # (query, fine) pair; no all-eligible fine contraction occurs.
            grad_fine_c = torch.empty_like(fine_c)
            _shared_grad_c[(bank_b * h * s, triton.cdiv(d, 32), triton.cdiv(d, 32))](
                channel, grad_output, order, starts, ends, grad_fine_c,
                q, h, m, s, d, b, bank_b, 32)
        return grad_channel, None, grad_fine_c


def selected_fine_numerator(channel, selected_id, fine_c):
    """Contract only selected fine C, preserving BF16 output and autograd."""
    if not torch.is_grad_enabled() and fine_c.shape[0] == channel.shape[0]:
        # Preserve the original inference contraction and launch pattern.
        b, q, h, m, d = channel.shape
        output = torch.empty((b, q, h, d), device=channel.device, dtype=channel.dtype)
        _selected_forward[(b * q * h, triton.cdiv(d, 32))](
            channel.contiguous(), selected_id.contiguous(), fine_c.contiguous(),
            output, q, h, m, fine_c.shape[2], d, 32)
        return output
    return _SelectedContraction.apply(channel, selected_id, fine_c)


@triton.jit
def _grouped_forward(A, C, ORDER, START, END, PAIRS,
                     Q: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                     S: tl.constexpr, D: tl.constexpr,
                     BANK_B: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    e = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    offsets = tl.arange(0, BLOCK)
    segment = group % S
    h = (group // S) % H
    b = group // (S * H)
    first, last = tl.load(START + group), tl.load(END + group)
    if first < last:
        c = tl.load(C + ((((b % BANK_B) * H + h) * S + segment) * D + d[:, None]) * D + e[None, :])
        for k in range(first, last, BLOCK):
            pos = k + offsets
            pair = tl.load(ORDER + (b * H + h) * Q * M + pos, pos < last, other=0)
            q, m = pair // M, pair % M
            row = ((b * Q + q) * H + h) * M + m
            a = tl.load(A + row[:, None] * D + d[None, :], pos[:, None] < last, other=0)
            contribution = tl.dot(a, c)
            tl.store(PAIRS + row[:, None] * D + e[None, :], contribution,
                     pos[:, None] < last)


@triton.jit
def _sum_selected(PAIRS, OUT, M: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    value = tl.full((D,), 0, tl.float32)
    for m in range(M):
        value += tl.load(PAIRS + (row * M + m) * D + d)
    tl.store(OUT + row * D + d, value)


@triton.jit
def _grouped_grad_a(C, G, ORDER, START, END, DA,
                    Q: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                    S: tl.constexpr, D: tl.constexpr,
                    BANK_B: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    e = tl.arange(0, D)
    offsets = tl.arange(0, BLOCK)
    segment = group % S
    h = (group // S) % H
    b = group // (S * H)
    first, last = tl.load(START + group), tl.load(END + group)
    if first < last:
        c = tl.load(C + ((((b % BANK_B) * H + h) * S + segment) * D + d[None, :]) * D + e[:, None])
        for k in range(first, last, BLOCK):
            pos = k + offsets
            pair = tl.load(ORDER + (b * H + h) * Q * M + pos, pos < last, other=0)
            q, m = pair // M, pair % M
            g = tl.load(G + ((b * Q + q[:, None]) * H + h) * D + e[None, :],
                        pos[:, None] < last, other=0)
            da = tl.dot(g, c)
            row = ((b * Q + q) * H + h) * M + m
            tl.store(DA + row[:, None] * D + d[None, :], da, pos[:, None] < last)


@triton.jit
def _shared_grad_c(A, G, ORDER, START, END, DC,
                   Q: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                   S: tl.constexpr, D: tl.constexpr,
                   B: tl.constexpr, BANK_B: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    e = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    offsets = tl.arange(0, BLOCK)
    segment = group % S
    h = (group // S) % H
    bank_b = group // (S * H)
    total = tl.full((BLOCK, BLOCK), 0, tl.float32)
    for b in range(bank_b, B, BANK_B):
        tile_group = (b * H + h) * S + segment
        first, last = tl.load(START + tile_group), tl.load(END + tile_group)
        acc = tl.full((BLOCK, BLOCK), 0, tl.float32)
        for k in range(first, last, BLOCK):
            pos = k + offsets
            pair = tl.load(ORDER + (b * H + h) * Q * M + pos, pos < last, other=0)
            q, m = pair // M, pair % M
            a = tl.load(A + ((((b * Q + q[None, :]) * H + h) * M + m[None, :])
                             * D + d[:, None]), pos[None, :] < last, other=0)
            g = tl.load(G + ((b * Q + q[:, None]) * H + h) * D + e[None, :],
                        pos[:, None] < last, other=0)
            acc += tl.dot(a, g)
        # Match r3's BF16 per-tile dC followed by a shared-bank sum.
        total += acc.to(tl.bfloat16).to(tl.float32)
    output = (((bank_b * H + h) * S + segment) * D + d[:, None]) * D + e[None, :]
    tl.store(DC + output, total)
