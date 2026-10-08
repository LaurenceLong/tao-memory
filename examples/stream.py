"""New-token-only prefill/decode and semantic snapshots; authorized CUDA only."""
import argparse
import torch
from common import ProjectedInputs, require_cuda
from tao_attention import new_state, StreamCache, tao_attention_prefill, tao_attention_decode
from tao_attention.presets import index_only_page_memo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    require_cuda()
    torch.manual_seed(42)
    # Graph's measured head geometry; no full language model is hidden here.
    model = ProjectedInputs(heads=6, width=128).cuda().eval()
    state = new_state(0, index_only_page_memo(graph=args.graph))
    snapshot_state = None
    try:
        with torch.inference_mode():
            # A real model obtains x from new token IDs and applies its positions.
            x = torch.randn(1, 2112, 768, device="cuda", dtype=torch.bfloat16)
            for start in range(0, 2048, 256):
                q, k, v, raw_q, raw_k = model(x[:, start:start+256])
                heads, state = tao_attention_prefill(q, k, v, raw_q=raw_q, raw_k=raw_k,
                    fusion_a=model.fusion_a, fusion_b=model.fusion_b, alpha=0.1,
                    state=state, position=state.get_pos())
            for index in range(2048, 2112):
                q, k, v, raw_q, raw_k = model(x[:, index:index+1])
                heads, state = tao_attention_decode(q, k, v, raw_q=raw_q, raw_k=raw_k,
                    fusion_a=model.fusion_a, fusion_b=model.fusion_b, alpha=0.1,
                    state=state, position=state.get_pos())
            snapshot_state = StreamCache.from_export(state.export())
            print({"position": state.get_pos(), "restored_position": snapshot_state.get_pos(),
                   "output_shape": tuple(heads.shape)})
            state.reset()
    finally:
        state.release_execution()
        if snapshot_state is not None:
            snapshot_state.release_execution()


if __name__ == "__main__":
    main()
