"""Caller-owned toy projections, not a pretrained model or training recipe."""
import torch
from torch import nn
from torch.nn import functional as F

class ProjectedInputs(nn.Module):
    def __init__(self, heads=2, width=64):
        super().__init__()
        self.heads, self.width = heads, width
        self.projection = nn.Linear(heads * width, 3 * heads * width, bias=False)
        self.fusion_a = nn.Parameter(torch.zeros(heads, width, 3))
        self.fusion_b = nn.Parameter(torch.zeros(heads, width, 3))

    def forward(self, x):
        projected = F.linear(x, self.projection.weight.to(x.dtype))
        raw_q, raw_k, v = projected.reshape(*x.shape[:2], 3, self.heads, self.width).unbind(2)
        # Real models apply their own positional transforms using absolute cursors.
        q = F.rms_norm(raw_q, (self.width,))
        k = F.rms_norm(raw_k, (self.width,))
        return q, k, v, raw_q, raw_k


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an authorized remote CUDA GPU; no CPU fallback")
