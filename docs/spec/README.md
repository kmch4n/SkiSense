# SkiSense: canonical engineering specifications

**Specification policy:** Accepted for all future repository changes.
**Implementation baseline:** `main@f95d68c2927de3fe8af5232fdd817f4d80a9380d` (inspected 2026-11-20).
**Evidence level:** Source-inspected; not a claim of successful model execution or scientific validation.

## Read this first

These English documents are the authoritative **description of the current system** and the required **specification-first development process**. They are designed to let a developer or coding agent understand the project's architecture, behavior, contracts, dependencies, and known deviations without inspecting every source file. Source code remains the final evidence of what actually executes; any disagreement is a defect to reconcile, never permission to silently invent expected behavior.

| File | Responsibility |
| --- | --- |
| [system-overview.md](system-overview.md) | Product scope, architecture, component ownership, mode matrix, data flow |
| [cli-and-configuration.md](cli-and-configuration.md) | CLI grammar, precedence, every configuration variable, device selection |
| [video-processing.md](video-processing.md) | Standard/fast video flows, tracking, target choice, per-frame state |
| [image-processing.md](image-processing.md) | Still-image path and differences from video |
| [pose-backends-and-data-contracts.md](pose-backends-and-data-contracts.md) | Backend factories, ROI/full-frame inference, landmark and topology contracts |
| [pose-scoring.md](pose-scoring.md) | Exact angle calculations, thresholds, missing data, score semantics |
| [zoom-and-rendering.md](zoom-and-rendering.md) | Zoom state machine, transforms, overlays and UI rendering |
| [model-comparison.md](model-comparison.md) | Comparison CLI, sampling, artifact/JSON schema, interpretation |
| [output-and-failure-semantics.md](output-and-failure-semantics.md) | Paths, generated artifacts, error paths and resource handling |
| [verification-matrix.md](verification-matrix.md) | Requirement identifiers, implementation owners, evidence and gaps |
| [change-protocol.md](change-protocol.md) | Binding document-first workflow, acceptance criteria and review requirements |

Read [system-overview.md](system-overview.md) for orientation; follow links into the relevant contract. To change behavior, begin at [change-protocol.md](change-protocol.md), **before editing code**.

## Authority and status

The **change protocol is normative now**. The remaining documents describe the **observed implementation baseline**, including current defects and limitations. An observation is *not* automatically an approved target design. When behavior is to change, propose and accept the new requirement in these documents *first*, then write tests and implementation.

Use these labels precisely in Issues, specs, and PRs:

- **Observed (as-is):** justified by a cited source path and behavior inspection; runtime reproduction may be absent.
- **Proposed (to-be):** suggested future contract, not yet accepted.
- **Accepted (to-be):** approved contract, ready for implementation.
- **Implemented:** implementation is present, but this alone does not prove its correctness.
- **Verified:** accompanied by specific reproducible test or real-model evidence, with scope, environment, and result.
- **Known deviation:** current code does not meet the intended behavior; link to an Issue rather than rewriting history.

This baseline was prepared from the tracked Python, tests, configuration, and documentation at the stated commit. It is not a runtime acceptance report. In particular, **0–100 is a heuristic relative feedback score, not a validated skiing grade**. SAM 3D Body produces estimated monocular 3D joints, not ground-truth angles. Never compare scores across backends without defining a common measurement and calibration procedure.

## Scope and source of truth

- Contract and module ownership: this directory.
- Developer and AI-agent obligations: [AGENTS.md](../../AGENTS.md), which must point here.
- User quick start: [README.md](../../README.md); historical Japanese introduction: [README_ja.md](../../README_ja.md).
- Operational references: [pose_backends.md](../pose_backends.md), [pose_comparison.md](../pose_comparison.md), [sam3d_setup.md](../sam3d_setup.md).
- Historical rationale and research are informational rather than authoritative specifications.
- Actual run behavior and measurable accuracy require tests. The [verification matrix](verification-matrix.md) records the present evidence boundaries.

## Rules for maintaining this suite

1. Every feature, behavior, default, public schema, error contract or processing change starts with a **specification change**. Do not edit executable code first and backfill the spec afterward.
2. Make the proposed outcome, edge cases, compatibility implications and acceptance criteria explicit before implementing.
3. Update the relevant named specification(s), traceability matrix and tests in the same PR. Do not rely on a README-only change.
4. Keep definitions in **one canonical location**: link to them elsewhere rather than maintaining competing copies.
5. Write all files under `docs/spec/` in **English only**; do not recreate `*_ja.md` specification duplicates. Existing Japanese user-facing/history documents may remain, with links redirected here.
6. Mark unverified inference behavior, scoring validity, platform compatibility and performance as unverified. Never promote code inspection or Python compilation into evidence of model accuracy.
7. If an existing defect blocks the proposed change, link the Issue, specify its acceptance criteria and separate remediation from unrelated refactoring.

No semantic version of the scoring algorithm, validated pose-quality ranking, reproducible model benchmark, or production support guarantee is currently established. Do not infer one from the package's `__version__ = "1.0.0"`.
