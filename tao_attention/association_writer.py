"""BF16 simplex features, FP32 association statistics and regularized solves.

The entrypoint retains IEEE FP32 CUDA matmul policy. Native autograd includes
the feature cast, statistics, trace regularizer and selected Cholesky API.
There is no inverse, detached coefficient, jitter or numerical fallback.
"""

import torch


def association_statistics(k, v, with_mass=False):
    """Raw K/V -> FP32 G/B matrices and optional true FP32 feature mass."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        u = k.float().softmax(dim=-1).to(torch.bfloat16)
        u = u.transpose(1, 2).float()
        values = v.transpose(1, 2).float()
        gram, cross = u.transpose(-1, -2) @ u, u.transpose(-1, -2) @ values
        return (gram, cross, u.sum(dim=-2)) if with_mass else (gram, cross)


def association_matrix(gram, cross, unchecked=False):
    """Solve independent [...,D,D] sufficient-statistic pairs, return BF16 C."""
    with torch.autocast(device_type=gram.device.type, enabled=False):
        width = gram.shape[-1]
        regularizer = gram.diagonal(dim1=-2, dim2=-1).sum(-1) / width
        system = gram + regularizer[..., None, None] * torch.eye(
            width, dtype=torch.float32, device=gram.device)
        if unchecked:
            factor, _ = torch.linalg.cholesky_ex(system, upper=False, check_errors=False)
        else:
            factor = torch.linalg.cholesky(system)
        return torch.cholesky_solve(cross, factor).to(torch.bfloat16)


def chronological_sum(statistics):
    """Sum fine axis -3 in FP32, matching streaming's first-fine initialization."""
    total = statistics.select(-3, 0)
    for index in range(1, statistics.shape[-3]):
        total = total + statistics.select(-3, index)
    return total


def feature_read_coefficients(h, read_z):
    """Scalar feature-mass mixture, then explicit BF16 query coefficients.

    h:[B,T,H,D], read_z broadcastable to [B,T,H,P,D].
    The contraction consumes these coefficients directly, without a denominator.
    """
    with torch.autocast(device_type=h.device.type, enabled=False):
        query = h.float().unsqueeze(-2)
        mass = (query * read_z.float().exp()).sum(-1)
        mixture = mass / mass.sum(-1, keepdim=True)
        return (mixture.unsqueeze(-1) * query).to(torch.bfloat16)
