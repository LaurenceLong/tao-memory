# Contributing

Keep the functional attention core independent of backbones, tokenizers and data
loaders. A focused change should describe its formula, tensor casts, causal
visibility, request ownership and compatibility impact. Avoid mixing numerical
changes with execution-only optimizations.

Run `python tools/check_source.py` for static syntax/metadata checks. Packaging checks can build distributions without importing Torch or running
attention. No automatic GitHub Actions workflow is enabled;
[docs/ci-workflow.yml](docs/ci-workflow.yml) is an opt-in reference template. Executable
model/kernel tests belong on an explicitly authorized remote CUDA GPU; never
silently substitute CPU validation. See [testing](docs/testing.md).

Include environment, exact source revision, input shape and execution mode with
GPU results. Report compilation/capture/setup separately from steady-state cost,
and include advancing-token seal/refresh events. Preserve unfavorable results.
Do not commit datasets, model weights, credentials, private host paths or logs.
Tests and benchmarks are evidence, not promises across untested hardware.

The MIT license applies to contributions; retain third-party notices. Discuss
broad API or numerical changes before a pull request. Public examples should use
caller-owned tensors rather than import a private training framework.
