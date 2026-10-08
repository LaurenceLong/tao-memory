"""Inference-only BF16 pointwise work around the unchanged Torch softmax."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _scores(Q, A, BIAS, SCORE,
            T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr, Q3: tl.constexpr,
            A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
            B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
             N: tl.constexpr, COARSE: tl.constexpr, FINE: tl.constexpr,
             LOCAL_BIAS: tl.constexpr,
            BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    d = index % D
    h = (index // D) % H
    t = (index // (D * H)) % T
    batch = index // (D * H * T)
    query = tl.load(Q + batch * Q0 + t * Q1 + h * Q2 + d * Q3,
                    valid, other=0).to(tl.float32)
    for branch in tl.static_range(3):
        coefficient = tl.load(A + h * A0 + d * A1 + branch * A2,
                              valid, other=0).to(tl.bfloat16,
                                                  fp_downcast_rounding="rtne").to(tl.float32)
        bias = tl.load(BIAS + h * B0 + d * B1 + branch * B2,
                       valid, other=0).to(tl.bfloat16,
                                           fp_downcast_rounding="rtne").to(tl.float32)
        product = (coefficient * query).to(tl.bfloat16,
                                           fp_downcast_rounding="rtne").to(tl.float32)
        score = (product + bias).to(tl.bfloat16,
                                     fp_downcast_rounding="rtne")
        if branch == 0:
            local_bias = tl.full((BLOCK,), LOCAL_BIAS, tl.float32).to(
                tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
            score = (score.to(tl.float32) + local_bias).to(
                tl.bfloat16, fp_downcast_rounding="rtne")
        if branch == 1 and not COARSE:
            score = tl.full((BLOCK,), -float("inf"), tl.float32).to(tl.bfloat16)
        if branch == 2 and not FINE:
            score = tl.full((BLOCK,), -float("inf"), tl.float32).to(tl.bfloat16)
        tl.store(SCORE + index * 3 + branch, score, valid)


@triton.jit
def _weighted_sum(RHO, LOCAL, COARSE, FINE, GAIN, OUT,
                  T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  R0: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr,
                  R3: tl.constexpr, R4: tl.constexpr,
                  L0: tl.constexpr, L1: tl.constexpr, L2: tl.constexpr, L3: tl.constexpr,
                  C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
                  F0: tl.constexpr, F1: tl.constexpr, F2: tl.constexpr, F3: tl.constexpr,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    d = index % D
    h = (index // D) % H
    t = (index // (D * H)) % T
    batch = index // (D * H * T)
    rho_offset = batch * R0 + t * R1 + h * R2 + d * R3
    local_offset = batch * L0 + t * L1 + h * L2 + d * L3
    coarse_offset = batch * C0 + t * C1 + h * C2 + d * C3
    fine_offset = batch * F0 + t * F1 + h * F2 + d * F3
    rho_l = tl.load(RHO + rho_offset, valid, other=0).to(tl.float32)
    rho_c = tl.load(RHO + rho_offset + R4, valid, other=0).to(tl.float32)
    rho_f = tl.load(RHO + rho_offset + 2 * R4, valid, other=0).to(tl.float32)
    local = tl.load(LOCAL + local_offset, valid, other=0).to(tl.float32)
    coarse = tl.load(COARSE + coarse_offset, valid, other=0).to(tl.float32)
    fine = tl.load(FINE + fine_offset, valid, other=0).to(tl.float32)
    gain = tl.load(GAIN).to(tl.float32)
    local_twice = (2.0 * local).to(tl.bfloat16,
                                    fp_downcast_rounding="rtne").to(tl.float32)
    coarse_scaled = (gain * coarse).to(tl.bfloat16,
                                        fp_downcast_rounding="rtne").to(tl.float32)
    fine_scaled = (gain * fine).to(tl.bfloat16,
                                    fp_downcast_rounding="rtne").to(tl.float32)
    local_term = (rho_l * local_twice).to(tl.bfloat16,
                                           fp_downcast_rounding="rtne").to(tl.float32)
    coarse_term = (rho_c * coarse_scaled).to(tl.bfloat16,
                                              fp_downcast_rounding="rtne").to(tl.float32)
    fine_term = (rho_f * fine_scaled).to(tl.bfloat16,
                                          fp_downcast_rounding="rtne").to(tl.float32)
    first_sum = (local_term + coarse_term).to(tl.bfloat16,
                                               fp_downcast_rounding="rtne").to(tl.float32)
    output = (first_sum + fine_term).to(tl.bfloat16,
                                         fp_downcast_rounding="rtne")
    tl.store(OUT + index, output, valid)


def fusion_scores(raw_q, coefficient, bias, coarse_present, fine_present):
    batch, length, heads, width = raw_q.shape
    count = raw_q.numel()
    score = torch.empty((*raw_q.shape, 3), device=raw_q.device, dtype=raw_q.dtype)
    local_bias = math.log(int(coarse_present) + int(fine_present)) if (
        coarse_present or fine_present) else 0.0
    _scores[(triton.cdiv(count, 256),)](
        raw_q, coefficient, bias, score, length, heads, width,
        *raw_q.stride(), *coefficient.stride(), *bias.stride(),
        count, coarse_present, fine_present, local_bias, 256,
        enable_fp_fusion=False)
    return score


def fusion_weighted_sum(rho, local, coarse, fine, gain):
    batch, length, heads, width = local.shape
    count = local.numel()
    output = torch.empty(local.shape, device=local.device, dtype=local.dtype)
    _weighted_sum[(triton.cdiv(count, 256),)](
        rho, local, coarse, fine, gain, output, length, heads, width,
        *rho.stride(), *local.stride(), *coarse.stride(), *fine.stride(),
        count, 256, enable_fp_fusion=False)
    return output
