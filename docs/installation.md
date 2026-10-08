# Installation

The historical measured environment is Linux, an NVIDIA RTX 4090, Python 3.12,
PyTorch 2.9.1 with CUDA 12.8 user-space libraries, and Triton 3.5.1. Driver/library
compatibility is the application's responsibility. Python 3.11 and newer are admitted by package metadata; static inspection found
no Python 3.12-only syntax or standard-library requirement. PyTorch 2.9.0 is also
within the dependency range. This source-compatibility statement is not a claim
of executed Python 3.11/Torch 2.9.0 validation or benchmark parity. Other CUDA
hardware and patch versions are not established by the supplied evidence. CPU, Windows,
MPS, ragged batches and quantized execution are not supported runtime claims.

Install a compatible CUDA PyTorch environment first, then from this repository:

```sh
python -m pip install .
```

For an already pinned environment, `python -m pip install --no-deps .` preserves
its dependency selection. The distribution is named **tao-memory**; the import is
**tao_attention**. It conflicts with earlier distributions that install the same
import package: use a clean environment rather than installing both together.
Triton kernels compile lazily on first real use and require writable compiler
caches. No datasets, models, tokenizer, compiler cache or CUDA toolkit is bundled.

An editable developer installation uses `python -m pip install -e .`.
The generic examples and GPU tests are source-distribution files, not pretrained
models or a reference training recipe. See [API](usage.md), [configuration](configuration.md)
and [Graph execution](graph.md) before running tensor workloads.
