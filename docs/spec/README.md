# SkiSense specification index

> Status: **Proposed** — baseline observed on `main` at `0fa4216f27bb907e5f99272831faff56ba42c2d0` (2026-11-18). Source inspection is not a substitute for ski-footage or GPU model validation. Approval is required before this baseline becomes normative.

## Purpose

Separate **as-is implementation**, **intended behavior**, **verified evidence**, and **historical decisions**. A feature being implemented does not imply that its accuracy, applicability, or safety has been validated.

| Document | Scope | Authority |
| --- | --- | --- |
| [behavior_ja.md](behavior_ja.md) | CLI, processing modes, backend selection, outputs and known defects | As-is baseline (proposed) |
| [scoring_ja.md](scoring_ja.md) | Joint-angle measurements, 100/50/0 score formula, comparability limits | As-is baseline (proposed) |
| [change_protocol_ja.md](change_protocol_ja.md) | Issue → spec → acceptance tests → implementation → PR | Proposed workflow |
| [verification_matrix_ja.md](verification_matrix_ja.md) | Requirement IDs, code location, tests, known gaps and quality gates | Traceability map (proposed) |
| [../project_details_ja.md](../project_details_ja.md) | Project history and rationale | Informational, not normative |
| [../pose_backends_ja.md](../pose_backends_ja.md) | Backend operations and trade-offs | Operational guidance |
| [../pose_comparison.md](../pose_comparison.md) | Experimental pose-model comparison | Research workflow |

## Status vocabulary

- **Proposed:** design or baseline awaiting agreement.
- **Accepted:** agreed contract before implementation.
- **Implemented:** source contains the behavior; it may still be unverified.
- **Verified:** objective test/inference evidence is recorded, with environment and scope.
- **Known issue:** code-level evidence suggests a contract defect; distinguish static inspection from reproduced runtime failures.

## Change rules

1. Create or update an Issue with reproducible current behavior and acceptance criteria.
2. Update relevant `docs/spec/` files in the **same PR** as any behavioral change.
3. Identify affected input type, normal/fast mode, backend, zoom, CLI and environment variables.
4. Define test evidence in advance. Distinguish static/mock tests from CUDA and actual skiing-video verification.
5. Changes to scoring, coordinate frames, units, model defaults, or emitted JSON must explicitly state comparability and compatibility.
6. Update README and operational docs after agreeing on behavior. Keep background/history truthful.

## Open quality gates

Known implementation issues: [#4](https://github.com/kmch4n/SkiSense/issues/4), [#12](https://github.com/kmch4n/SkiSense/issues/12), [#15](https://github.com/kmch4n/SkiSense/issues/15), [#16](https://github.com/kmch4n/SkiSense/issues/16), [#17](https://github.com/kmch4n/SkiSense/issues/17). Controlled evaluation and CI: [#3](https://github.com/kmch4n/SkiSense/issues/3), [#9](https://github.com/kmch4n/SkiSense/issues/9), [#18](https://github.com/kmch4n/SkiSense/issues/18), [#19](https://github.com/kmch4n/SkiSense/issues/19).

The present spec does **not** claim scientific validation or production readiness. Current implementation details are a versioned baseline, not a promise that every documented option behaves as its name suggests. See [SAM3D setup](../sam3d_setup.md).
