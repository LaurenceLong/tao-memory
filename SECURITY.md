# Security and operational safety

Do not deserialize untrusted PyTorch state snapshots or execute untrusted model
code. Exported request states contain tensors and are intended for trusted,
same-device process use; they are not a portable or authenticated file format.

Custom Triton kernels assume the documented dimensions, dtype and causal input
contract. This alpha implementation is not a hardened service boundary for
arbitrary user-controlled shapes. Isolate requests and enforce input limits in
the serving application. CUDA Graph state must not be shared or mutated
concurrently.

For a suspected vulnerability, use the repository's private vulnerability
reporting feature if enabled. Otherwise contact a maintainer privately before
posting exploit details. This project does not promise an emergency response SLA.
Never include secrets, private prompts or model weights in a public report.
