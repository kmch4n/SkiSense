## Summary

Describe the problem, user-visible behavior, and linked Issue(s).

## Specification / Contract

- [ ] Updated `docs/spec/` (or explained why no observable contract changes)
- [ ] Distinguished as-is behavior from expected behavior and validation evidence
- [ ] Updated README / backend docs / CLI help where relevant
- [ ] Checked video/image, normal/fast, zoom-on/off, YOLO/SAM3D applicability

## Changes

Describe implementation, documentation, migration and compatibility impacts.

## Verification

- [ ] `python -m unittest discover -s tests -v` (where dependencies are available)
- [ ] `python -m unittest discover -s tests -p 'test_docs_links.py' -v`
- [ ] Syntax/static checks
- [ ] Real CPU/MPS/CUDA/model-weight runs (list only those actually performed)

State exact commands, runtime/environment, results, and **tests not performed**. Do not claim unexecuted tests passed.

## Risks / Known gaps

Regressions, uncertain model semantics, performance implications and follow-up Issues.

## Acceptance criteria

- [ ] Linked Issue requirements met, or deferred explicitly with rationale.
