## Summary

Explain the problem, the agreed outcome, observable differences and linked Issue(s).

## Specification-first contract (required for behavior changes)

- [ ] Read `AGENTS.md` and `docs/spec/README.md` before changing code.
- [ ] Updated the affected canonical **English** `docs/spec/` files **before executable changes** (or documented why this PR is strictly nonbehavioral).
- [ ] Distinguished current behavior, accepted target behavior, known deviations and verification evidence.
- [ ] Defined acceptance criteria and edge cases across applicable input/mode/backend/zoom/target settings.
- [ ] Updated `docs/spec/verification-matrix.md` for modified contracts.
- [ ] Documented changes to API/CLI/config, model data shape, coordinate units, output schema, scoring comparability and backward compatibility (or stated none).

**Specification sections changed first:**
**Affected contract IDs:**
**Behavioral compatibility / migration:**

## Changes

Describe the implementation, documentation, tests and why the diff is minimal and maintainable.

## Verification

- [ ] Python syntax: `python -m compileall -q run.py compare.py src tests`
- [ ] Relative Markdown links: `python -m unittest discover -s tests -p 'test_docs_links.py' -v`
- [ ] Broader unit tests: `python -m unittest discover -s tests -v` (if dependencies available)
- [ ] Relevant CPU/mock regression and real image/video/model tests (state exact coverage, if applicable)

**Executed commands, environment and exact results:**
**Tests NOT run and why:**
**Unverified behavior / hardware / model quality:**

Do not state that tests passed unless actually executed. Static checks do not establish runtime inference or domain validity.

## Risks, known limitations and follow-up

List effects on accuracy, memory, latency, IO, license/dependencies, output compatibility and known defects. Link related Issues rather than silently broadening scope.

## Acceptance criteria

- [ ] All agreed acceptance criteria met, or each incomplete criterion explicitly tracked and justified.
