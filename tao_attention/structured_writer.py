"""FP32 sufficient statistics and analytic structured-Gram association."""
import torch


def structured_statistics(k, v):
    """Return q:[B,H,D], B:[B,H,D,D], m:[B,H,D], never dense Gram."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        u = k.float().softmax(dim=-1).to(torch.bfloat16).transpose(1, 2).float()
        values = v.transpose(1, 2).float()
        mass = u.sum(-2)
        square_mass = (u * u).sum(-2)
        cross = u.transpose(-1, -2) @ values
        return square_mass, cross, mass


def structured_matrix(square_mass, cross, mass, count):
    """Apply the frozen diagonal/two-rank inverse in FP32; final C is BF16."""
    with torch.autocast(device_type=cross.device.type, enabled=False):
        variance = (square_mass - mass * mass / count).clamp_min(0)
        variance_sum = variance.sum(-1, keepdim=True)
        ridge = square_mass.sum(-1, keepdim=True) / mass.shape[-1]
        diagonal = variance + ridge
        t0 = variance / diagonal
        zero = variance_sum == 0
        t = torch.where(zero, torch.zeros_like(t0), t0)
        positive_sum = (variance * ridge / diagonal).sum(-1, keepdim=True)
        safe_sum = torch.where(zero, torch.ones_like(positive_sum), positive_sum)
        t_cross = (t.unsqueeze(-1) * cross).sum(-2)
        hb = cross / diagonal.unsqueeze(-1) + t.unsqueeze(-1) * t_cross.unsqueeze(-2) / safe_sum.unsqueeze(-1)
        hm = mass / diagonal + t * (t * mass).sum(-1, keepdim=True) / safe_sum
        mass_hb = (mass.unsqueeze(-1) * hb).sum(-2)
        denominator = count + (mass * hm).sum(-1, keepdim=True)
        result = hb - hm.unsqueeze(-1) * mass_hb.unsqueeze(-2) / denominator.unsqueeze(-1)
        return result.to(torch.bfloat16)
