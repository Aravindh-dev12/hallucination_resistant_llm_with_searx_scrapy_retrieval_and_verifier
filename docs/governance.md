# Governed model-resource architecture

External projects are resources, not implicit components of the sovereign
checkpoint. VibeThinker, ThinkCoder, Prime RL, Prime Verifiers, ShinkaEvolve,
Monty and Megatron-LM may contribute only provenance-accepted,
license-compatible and verifier-backed training records.

V-JEPA 2.1 is registered as a governed perception service. LeWM is registered
as a governed action-conditioned world-state service. Aegis, memory, consent,
formal verification, sandbox execution and audit receipts remain OEON kernel
services.

Tinton is an unprivileged coordinator. It can propose and route candidates but
cannot authorize itself, count itself toward an independent quorum, bypass a
kernel gate, silently fuse external code or weights, or promote a checkpoint.

## Enforcement

- Default-deny resource registry: `config/governed_resources.yaml`
- Executable admission and promotion gates: `src/governance.py`
- Boundary and positive-path tests: `tests/test_governance.py`

Every accepted training resource must be pinned by version and immutable
content digest. Production deployments should persist signed audit receipts in
append-only storage and verify signatures at every service boundary.
