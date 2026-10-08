"""Inference selected-fine read with the original Torch denominator."""

import torch
import triton
import triton.language as tl


@triton.jit
def _query_fresh_selected_fine(HV, PZ, FZ, IDS, C, CHANNEL, NUM,
          Q: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
          S: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
          HS0: tl.constexpr, HS1: tl.constexpr, HS2: tl.constexpr,
          HS3: tl.constexpr, HS4: tl.constexpr,
          PS0: tl.constexpr, PS1: tl.constexpr, PS2: tl.constexpr,
          PS3: tl.constexpr,
          ZS0: tl.constexpr, ZS1: tl.constexpr, ZS2: tl.constexpr,
          ZS3: tl.constexpr,
          IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
          IS3: tl.constexpr,
          CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr,
          CS3: tl.constexpr, CS4: tl.constexpr,
          E_BLOCK: tl.constexpr, M_BLOCK: tl.constexpr,
          COARSE_GROUP: tl.constexpr):
    row = tl.program_id(0)
    e = tl.program_id(1) * E_BLOCK + tl.arange(0, E_BLOCK)
    d = tl.arange(0, D)
    h = row % H
    q = (row // H) % Q
    b = row // (H * Q)
    acc = tl.full((E_BLOCK,), 0, tl.float32)
    for m in range(M):
        segment = tl.load(IDS + b * IS0 + q * IS1 + h * IS2 + m * IS3)
        page = segment // COARSE_GROUP
        z = tl.load(FZ + b * ZS0 + h * ZS1 + segment * ZS2 + d * ZS3).to(tl.float32)
        pz = tl.load(PZ + b * PS0 + h * PS1 + page * PS2 + d * PS3).to(tl.float32)
        hv = tl.load(HV + b * HS0 + q * HS1 + h * HS2 + page * HS3 + d * HS4).to(tl.float32)
        # Torch BF16 subtraction and exp each materialize before multiplication.
        difference = (z - pz).to(tl.bfloat16).to(tl.float32)
        segment_weight = tl.exp(difference).to(tl.bfloat16).to(tl.float32)
        channel = (hv * segment_weight).to(tl.bfloat16).to(tl.float32)
        if tl.program_id(1) == 0:
            tl.store(CHANNEL + (row * M + m) * D + d,
                     channel.to(tl.bfloat16))
        c = tl.load(C + b * CS0 + h * CS1 + segment * CS2 +
                    d[:, None] * CS3 + e[None, :] * CS4,
                    e[None, :] < D, other=0).to(tl.float32)
        acc += tl.sum(channel[:, None] * c, 0)
    numerator = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(NUM + row * D + e, numerator.to(tl.bfloat16), e < D)


def query_fresh_fine_parts(hv, page_z, fine_z, selected_id, fine_c,
                           coarse_group=8):
    b, q, h, p, d = hv.shape
    m = selected_id.shape[-1]
    selected_channel = torch.empty((b, q, h, m, d), device=hv.device,
                                   dtype=hv.dtype)
    numerator = torch.empty((b, q, h, d), device=hv.device, dtype=hv.dtype)
    _query_fresh_selected_fine[(b * q * h, triton.cdiv(d, 32))](
        hv, page_z, fine_z, selected_id, fine_c,
        selected_channel, numerator,
        q, h, p, fine_z.shape[2], m, d,
        *hv.stride(), *page_z.stride(), *fine_z.stride(),
        *selected_id.stride(), *fine_c.stride(),
        32, triton.next_power_of_2(m), coarse_group,
        enable_fp_fusion=False)
    denominator = selected_channel.sum(-1).sum(-1, keepdim=True).to(hv.dtype)
    fine = (numerator / denominator).to(hv.dtype)
    return selected_channel, numerator, denominator, fine


def query_fresh_fine_read(hv, page_z, fine_z, selected_id, fine_c,
                          coarse_group=8):
    return query_fresh_fine_parts(
        hv, page_z, fine_z, selected_id, fine_c, coarse_group)[-1]
