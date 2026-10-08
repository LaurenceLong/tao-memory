"""One caller-owned update; run only on an authorized remote CUDA GPU."""
import torch
from common import ProjectedInputs, require_cuda
from tao_attention import tao_attention
from tao_attention.presets import index_only_page_memo


def main():
    require_cuda()
    torch.manual_seed(42)
    model = ProjectedInputs().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(1, 1024, 128, device="cuda", dtype=torch.bfloat16)
    optimizer.zero_grad(set_to_none=True)
    q, k, v, raw_q, raw_k = model(x)
    heads = tao_attention(q, k, v, raw_q=raw_q, raw_k=raw_k,
        fusion_a=model.fusion_a, fusion_b=model.fusion_b, alpha=0.1,
        config=index_only_page_memo())
    # Replace this illustrative scalar with your output projection and task loss.
    loss = heads.float().square().mean()
    loss.backward()
    optimizer.step()
    print({"loss": loss.detach().item(), "shape": tuple(heads.shape)})


if __name__ == "__main__":
    main()
