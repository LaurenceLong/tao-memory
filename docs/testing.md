# Validation

## Local static and packaging checks

`python tools/check_source.py` parses Python and packaging metadata without
importing Torch/Triton. To build sdist/wheel and check metadata without model or
kernel execution, run:

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
```

No automatic GitHub Actions workflow is enabled. The optional
[workflow template](ci-workflow.yml) can be copied to `.github/workflows/` by a
repository maintainer with the necessary permissions. It performs only the above
static/package checks and does not claim GPU correctness. This export's added examples
and tests have only been syntax checked locally.

## Authorized remote CUDA GPU only

```sh
python -m pip install -e .
python tests/gpu_sanity.py
python examples/train.py
python examples/stream.py --graph
```

These opt-in checks cover a real differentiable causal sequence, advancing state,
export/import/reset and eager/Graph request isolation. They require an authorized
CUDA machine and fail rather than fall back to CPU. They are developer regression
checks, not a full training recipe, performance benchmark or obligatory experiment
preflight. The numeric tolerances in tests are test-specific, not a performance
or publication acceptance threshold. Report exact hardware/software and failures.
For performance, include setup/capture and refresh/seal events; do not benchmark
replay on frozen queries or compare unaligned execution contexts.

## Runtime compatibility evidence

Package metadata admits Python 3.11 and newer and PyTorch 2.9.x. Static parsing
with Python 3.11 grammar does not establish CUDA/kernel compatibility. Historical
measurements used Python 3.12 and Torch 2.9.1; a new runtime must report its own
actual sanity outcome without relabeling those measurements. The existing
`tests/gpu_sanity.py` is unchanged and remains opt-in on an authorized remote GPU.
