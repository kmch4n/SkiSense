# Specification-to-code-to-evidence matrix

**Status:** Source-inspection baseline at `main@f95d68c` (2026-11-20).
**Evidence vocabulary:** `Code` = supported by reading implementation; `Unit` = named test exists but has **not been executed in this documentation review**; `CI` = configured static workflow (not proof of a successful current run); `Model` = actual model/hardware evidence (not established here); `Gap` = missing requirement or known defect.

This document maps important behavioral claims to ownership, existing evidence and remaining acceptance work; it **does not assert that all tests passed**. The named test suite is `unittest`, but the existing CI job runs Python compilation and only `test_docs_links.py`.

| ID | Observed contract or intended correction | Main implementation | Existing evidence / status | Follow-up |
| --- | --- | --- | --- | --- |
| SYS-01 | Local video/image/compare pipelines, no server | `run.py`, `compare.py`, `src/skisense/` | Code | Validate supported host setup |
| CLI-01 | Explicit backend > env > `yolo26` | `config.py`, `backends/__init__.py`, `run.py` | Code; `tests/test_backend_selection.py` exists | Runtime package/device selection |
| CLI-02 | `--high` currently logs only, no interpolation | `run.py`, `main.py` | Code; known deviation | [#4](https://github.com/kmch4n/SkiSense/issues/4) |
| CFG-01 | Config rooted at source path; env type validation/default fallback | `config.py` | Code | Test parsing, invalid values and external CWD |
| DEV-01 | Auto MPS > CUDA > CPU; SAM3D requires CUDA | `main.py`, `sam3d_backend.py` | Code; no GPU evidence | Device-specific integration |
| VID-01 | Normal video: YOLOv8x → per-bbox pose → target match → overlay | `main.py` | Code; no orchestrator tests | [#15](https://github.com/kmch4n/SkiSense/issues/15), integration |
| VID-02 | Longest track chosen by frames seen, mean area and center distance | `main.py` | Code | Mock pre-scan/tracker tests |
| VID-03 | Fast YOLO pose does full-frame inference; longest/zoom pre-scan remains | `main.py`, `yolo11_backend.py` | Code | [#9](https://github.com/kmch4n/SkiSense/issues/9) |
| VID-04 | Fast SAM3D has no configured detector; full-frame fallback not reliable for multiple people | `sam3d_backend.py`, `main.py` | Code; upstream-dependent behavior unverified locally | [#16](https://github.com/kmch4n/SkiSense/issues/16) |
| VID-05 | Normal zoom-off picks first score and skips target skeleton/bbox; active `largest` still updates Deep SORT but does not use its tracks | `main.py` | Code; known deviation | [#15](https://github.com/kmch4n/SkiSense/issues/15) |
| IMG-01 | Largest-person still image, bbox if pose missing, identity coordinates | `image_processor.py` | Code | Multi-person, no-person, corrupt-file tests |
| BACK-01 | YOLO11/26 COCO-17 normalization and ROI person association | `yolo11_backend.py`, `pose_topology.py` | Code; `tests/test_pose_comparison.py` includes ROI association | Real footage, edge cases |
| BACK-02 | SAM3D 63 MHR landmarks, 3D estimates, synthetic visibility | `sam3d_backend.py`, `pose_topology.py` | Code; no current CUDA execution evidence | Gated model integration, occlusions |
| SCORE-01 | Angle evaluators and 100/50/0 mean over available metrics | `pose_analyzer.py` | Code; no dedicated scoring-boundary tests | [#19](https://github.com/kmch4n/SkiSense/issues/19) |
| SCORE-02 | Shoulder line raw atan2 may wrap near 180° | `pose_analyzer.py` | Code; known deviation | [#12](https://github.com/kmch4n/SkiSense/issues/12) |
| SCORE-03 | COCO ankle N/A; 0.0 output sentinel distinct from grade | `pose_analyzer.py`, `main.py` | Code | Missing-data/boundary test |
| ZOOM-01 | Immediate area-based crop; reuse last target up to 29 missing frames | `zoom_tracker.py` | Code | Edge crop and timeout tests |
| ZOOM-02 | Rendering transforms ROI→source→crop and uses per-topology edges | `main.py` | Code | Deterministic image/zoom tests |
| CMP-01 | Samples same frames, largest detector target, sequential backends | `pose_comparison.py` | Code; `tests/test_pose_comparison.py` exists | End-to-end comparison footage |
| CMP-02 | RTMW3D projected 2D comparison only; no score | `compare_rtmw3d.py` | Code; `tests/test_compare_rtmw3d.py` exists | 3D coordinate validation |
| CMP-03 | Comparison JSON records metadata, selected bbox, per-model statuses and estimate-call timing (including lazy RTMW3D setup when triggered) | `pose_comparison.py`, `compare_rtmw3d.py` | Code; serialization unit tests exist | Schema snapshot and timing-boundary tests |
| OUT-01 | Video writer should fail visibly when encoding unavailable | `main.py` | **Gap:** no `isOpened()` or export verification | [#17](https://github.com/kmch4n/SkiSense/issues/17) |
| OUT-02 | Still-image writer, input copy, resources and exit codes | `image_processor.py`, `main.py` | Code; partial cleanup/write checks | Add exception/write-failure tests |
| DOC-01 | Relative Markdown links are checked automatically | `tests/test_docs_links.py`, `.github/workflows/static-checks.yml` | CI configured; no runtime imports | Anchor links and drift checks not yet covered |
| TEST-01 | Core video/zoom/scoring behavior regression coverage | `tests/`, `.github/workflows/static-checks.yml` | **Gap:** CI does not run existing model-dependent unit tests | Originally [#18](https://github.com/kmch4n/SkiSense/issues/18); **Issue currently closed despite remaining scope** |
| PROC-01 | Specification updated **before** runtime behavior changes | `AGENTS.md`, `docs/spec/change-protocol.md`, PR template | Policy established here; manual review required | Add enforceable PR drift checks if valuable |

## Existing test commands

```bash
# Current GitHub Actions static job:
python -m compileall -q run.py compare.py src tests
python -m unittest discover -s tests -p 'test_docs_links.py' -v

# Broader local unit suite; requires project ML dependencies:
python -m unittest discover -s tests -v
```

The current tests cover backend factory selection and mock YOLO checkpoint construction, target matching in a YOLO ROI, comparison frame index/target selection, serialization and draw behavior, RTMW3D association, plus relative Markdown links. They **do not** exercise real video orchestration, the full scoring boundary matrix, zoom timeouts or real model accuracy. No full suite, CUDA model run, media-generation smoke test or expert-judged footage validation was executed while authoring this baseline.

## Traceability maintenance

When changing a contract, preserve the ID, revise its statement, specify new expected results and update its evidence status **after actual execution**. Add IDs for new public behavior; do not mark `Verified` from a mere source-code scan, CI configuration or historical Issue closure. Keep the issue state and remaining work distinct: a closed tracker item does not erase an untested requirement. See [change-protocol.md](change-protocol.md).
