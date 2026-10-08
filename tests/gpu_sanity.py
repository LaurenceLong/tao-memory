"""Opt-in standalone CUDA sanity, never executed by static CI or on CPU.

Run only on an explicitly authorized remote GPU: python tests/gpu_sanity.py
This finite check is not a training recipe or performance measurement.
"""
import torch
from tao_attention import tao_attention, tao_attention_prefill, tao_attention_decode
from tao_attention import new_state, StreamCache
from tao_attention.presets import index_only_page_memo


def operands(tokens, requires_grad=False):
    return [torch.randn(1, tokens, 6, 128, device="cuda", dtype=torch.bfloat16,
                        requires_grad=requires_grad) for _ in range(5)]


def invoke(fn, tensors, a, b, **kwargs):
    q, k, v, raw_q, raw_k = tensors
    return fn(q, k, v, raw_q=raw_q, raw_k=raw_k, fusion_a=a, fusion_b=b,
              alpha=0.1, **kwargs)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized CUDA GPU required; no CPU fallback")
    torch.manual_seed(42)
    a = torch.zeros(6, 128, 3, device="cuda", requires_grad=True)
    b = torch.zeros_like(a, requires_grad=True)
    inputs = operands(1024, requires_grad=True)
    output = invoke(tao_attention, inputs, a, b, config=index_only_page_memo())
    output.float().square().mean().backward()
    assert output.shape == inputs[0].shape and torch.isfinite(output).all()
    for tensor in [*inputs, a, b]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    del inputs, output
    torch.cuda.synchronize()
    results = []
    # Identical actual token operands, independent requests, same CUDA stream.
    with torch.inference_mode():
        data = operands(1089)
        for graph in (False, True):
            state = new_state(0, index_only_page_memo(graph=graph))
            restored = None
            try:
                for start in range(0, 1024, 256):
                    _, state = invoke(tao_attention_prefill,
                        [x[:, start:start+256] for x in data], a, b,
                        state=state, position=state.get_pos())
                outputs = []
                for index in range(1024, 1088):
                    value, state = invoke(tao_attention_decode,
                        [x[:, index:index+1] for x in data], a, b,
                        state=state, position=state.get_pos())
                    outputs.append(value.clone())
                assert state.get_pos() == 1088
                snapshot = state.export()
                restored = StreamCache.from_export(snapshot)
                token = [x[:, 1088:1089] for x in data]
                left, state = invoke(tao_attention_decode, token, a, b,
                                    state=state, position=state.get_pos())
                right, restored = invoke(tao_attention_decode, token, a, b,
                                    state=restored, position=restored.get_pos())
                # Test-specific BF16 tolerance; this is not a benchmark gate.
                torch.testing.assert_close(left, right, atol=1e-2, rtol=1e-2)
                result = torch.cat(outputs, dim=1)
                assert torch.isfinite(result).all()
                results.append(result)
                state.reset()
                assert state.get_pos() == 0 and restored.get_pos() == 1089
            finally:
                state.release_execution()
                if restored is not None:
                    restored.release_execution()
        torch.testing.assert_close(results[0], results[1], atol=1e-2, rtol=1e-2)
    torch.cuda.synchronize()
    print("PASS: differentiable training, advancing eager/Graph, export/import/reset")


if __name__ == "__main__":
    main()
