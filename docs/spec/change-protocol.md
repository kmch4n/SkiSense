# Specification-first engineering and change-control protocol

**Policy status: ACCEPTED.** Applies to human developers and all AI coding agents working on SkiSense. **Authority:** This document, [README.md](README.md), and [AGENTS.md](../../AGENTS.md).

## 1. Non-negotiable order

**Read → Define/approve specification → Define acceptance tests → Implement → Verify → Review → Merge.**

A feature, bug fix, default change, output change, performance-mode change or compatibility-affecting refactor must **begin by editing the relevant `docs/spec/` Markdown files**. The agent must not modify executable source files and then retroactively invent documentation to explain them. The spec change must be reviewable as the first logical stage, even when a final PR contains both spec and code. If an agent is instructed to implement without enough product decisions, record the ambiguity explicitly and use conservative assumptions; do not silently rewrite the intended contract.

## 2. Stage gates

### Gate 0: Understand current reality

Read this index and relevant specs; inspect actual implementations, tests, `AGENTS.md`, active Issues/PRs and recent commits. Reconcile whether the request concerns as-is behavior, proposed to-be behavior, or a known defect. Preserve the source baseline SHA. Never present a proposal as current behavior or interpret a closed Issue as verified without evidence.

### Gate 1: Write the specification **before code**

Edit the smallest authoritative set under `docs/spec/`, adding a new topic file only when ownership is clear. The change must state:

- Motivation and expected user-visible behavior; explicit non-goals;
- Public entry points, inputs/outputs, data shapes, units, coordinate systems and invariants;
- Behavior across **video/image, normal/fast, zoom on/off, longest/largest, YOLO/SAM3D**, where applicable;
- Boundary cases, missing data, error modes, resource lifecycle and fallback;
- CLI, `.env`, model/checkpoint, weight licensing, platform and output/JSON compatibility;
- Deterministic acceptance criteria, proposed tests and necessary real-model or real-ski-video evidence;
- Comparative interpretation, scoring-version/methodology impacts, and migration/rollback needs.

Describe the expected change as **Proposed (to-be)** until accepted; distinguish it from **Observed (as-is)**. If the change is authorized and requirements are unambiguous, mark the to-be contract **Accepted**. If requirements are contradictory, resolve the contradiction in the spec before implementation.

### Gate 2: Tests and evidence plan

Specify expected behavior before adapting tests to match implementation. Use a proportionate test pyramid:

1. Static: syntax, Markdown relative links, configuration and contract references.
2. Deterministic unit/mock: threshold boundaries, target association, zoom transforms, missing outputs, invalid input and resource cleanup.
3. Integration: representative video/image runs on supported CPU/MPS/CUDA combinations with installed weights.
4. Domain validation: multiple ski views, phases, occlusions, ability levels, expert annotations and repeatability for any accuracy claim.

A test that was **not executed** is not a passed test. Compilation and link checks do not establish model execution or scientific validity.

### Gate 3: Implementation

Implement exactly the accepted contract, keeping the diff focused and backwards compatible where required. If code inspection reveals new requirements or contradictions, **stop that implementation path and amend the specification first**. Prefer deterministic helpers and explicit failure handling to unreviewable orchestration changes. Do not mix speculative benchmark conclusions into a fix.

### Gate 4: Synchronize, verify, review

Update `verification-matrix.md` IDs, README/operational docs and `.env.example`/CLI help when affected. Compare the final diff against the acceptance criteria. Run all feasible tests and record exact commands/results, OS/device/dependency scope, not-run tests and known regressions. Check that no stale links or obsolete `*_ja.md` spec references remain.

The PR body must follow [the repository template](../../.github/pull_request_template.md), link the relevant Issue(s), identify spec sections updated **before implementation**, enumerate test evidence, and state changes to semantics/compatibility. Request review; never claim readiness on the basis of static checks alone.

## 3. Review and merge criteria

A PR is **not merge-ready** if it silently changes a public behavior without updating specs, lacks acceptance criteria for a behavior change, asserts hardware/accuracy validation without evidence, loses track of a known defect, or fails required available checks. Reviewer approval must establish coherence among intended spec, implementation, tests and user docs. Do not merge a major behavior change merely because a bot reported green syntax checks.

Commit history must follow existing repository conventions: **English, coherent, purpose-driven commits**, subject format such as `[📝] Document ...`, `[✨] Add ...`, `[🐛] Fix ...`, or `[✅] Test ...` when applicable. Avoid a commit per file or iterative AI-chatter commits; use one cohesive commit or a few independently understandable stages (for example, accepted spec, implementation/tests). Match the existing repo's actual recent commit style. PR title/body are English and professional. Prefer squash merge when many granular development commits would pollute `main` history.

## 4. Documentation-only and exceptional changes

A documentation-only PR may update the observed baseline and process without modifying runtime code. It still requires source cross-checks, link checks and explicit verification limitations. Spelling fixes and other truly nonbehavioral edits need not introduce a false feature specification, but must not silently invalidate existing contracts. Emergency production fixes still require documenting the intended changed behavior **before** code, even if the document and code ship in one reviewed PR.

English is mandatory for all files in `docs/spec/`. Existing historical Japanese project narratives and user-facing translations are informational and may remain; they must link to the canonical English spec instead of maintaining divergent normative definitions.

## 5. Status transitions and disagreements

A requirement progresses `Proposed → Accepted → Implemented → Verified`, with evidence specific to its scope. It may remain implemented but unverified, or known-deviating with an open Issue. If a source implementation and approved to-be specification differ, do **not** rewrite the specification to rationalize the bug. Document the observed departure, link the Issue and address it with explicit regression tests. Repository-wide source inspection, static checks and thin mocks are valuable but never substitute for model/hardware testing and domain-level validation.
