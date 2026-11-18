# 仕様・実装・検証の対応表

- **基準:** `main@0fa4216f`（2026-11-18）と本ドキュメント整備PR #20
- **状態:** Proposed。静的コード確認を中心としたマトリクスであり、実映像検証の完了を示すものではない
- **定義:** 「静的」はソース上の事実確認、「自動」はCIまたは再実行可能なテスト、「実機」はモデルを用いた映像評価を意味する

| ID | 仕様・論点 | 対象コード / テスト | 検証と残課題 |
| --- | --- | --- | --- |
| CLI-01 | 既定はYOLO26、CLIフラグが環境変数に優先 | [config.py](../../src/skisense/config.py), [backends/__init__.py](../../src/skisense/backends/__init__.py), [test_backend_selection.py](../../tests/test_backend_selection.py) | 実装確認、既存単体テストあり。PR #20では全単体テスト未実行 |
| CLI-02 | `--high` は受理されるが補間は未実装 | [run.py](../../run.py), [main.py](../../src/skisense/main.py) | **既知の不一致**: [#4](https://github.com/kmch4n/SkiSense/issues/4) |
| VID-01 | 通常動画は人物bboxを検出し選択人物の骨格・スコアを表示 | [main.py](../../src/skisense/main.py) | ズーム無効時の描画/対象選択バグ [#15](https://github.com/kmch4n/SkiSense/issues/15)。回帰テスト未追加 |
| VID-02 | 高速YOLO系は全画面から各人物の姿勢を返す | [main.py](../../src/skisense/main.py), [yolo11_backend.py](../../src/skisense/backends/yolo11_backend.py) | 実装確認。代表的なスキー映像の実機評価・ROI比較は [#9](https://github.com/kmch4n/SkiSense/issues/9) |
| VID-03 | 高速SAM3Dは現状検出器なし、単一画像全体bboxで推論 | [sam3d_backend.py](../../src/skisense/backends/sam3d_backend.py), [導入手順](../sam3d_setup.md) | 上流コードと照合。対象人物検出の未達は [#16](https://github.com/kmch4n/SkiSense/issues/16)、CUDA実機試験なし |
| VID-04 | `longest`は選択trackを事前走査 | [main.py](../../src/skisense/main.py), [zoom_tracker.py](../../src/skisense/zoom_tracker.py) | ズーム有効時だけの現行条件を確認。[#15](https://github.com/kmch4n/SkiSense/issues/15)で修正対象 |
| OUT-01 | MP4書き出しは成功可否を確認して報告する | [main.py](../../src/skisense/main.py) | **未達**: `VideoWriter.isOpened()`なし。[#17](https://github.com/kmch4n/SkiSense/issues/17) |
| SCORE-01 | 0–100は測定可能項目の100/50/0平均 | [pose_analyzer.py](../../src/skisense/pose_analyzer.py), [スコア仕様](scoring_ja.md) | 現行式を記述。妥当性未検証 [#19](https://github.com/kmch4n/SkiSense/issues/19) |
| SCORE-02 | 肩の水平角は左右点の並びに依らず正しく評価 | [pose_analyzer.py](../../src/skisense/pose_analyzer.py) | **未達**: atan2折り返しの問題 [#12](https://github.com/kmch4n/SkiSense/issues/12) |
| COMP-01 | 同一フレームで複数モデルの投影を比較しスコアを混同しない | [pose_comparison.py](../../src/skisense/pose_comparison.py), [test_pose_comparison.py](../../tests/test_pose_comparison.py), [test_compare_rtmw3d.py](../../tests/test_compare_rtmw3d.py) | コード・既存テストを確認。実際のモデル間優劣は未確定 [#3](https://github.com/kmch4n/SkiSense/issues/3) |
| DOC-01 | Markdown相対リンクとPython構文の静的検査 | [test_docs_links.py](../../tests/test_docs_links.py), [CI](../../.github/workflows/static-checks.yml) | PR #20上のGitHub Actionsで成功を確認。GPU検証・既存全単体テストは対象外 |
| TEST-01 | 動画制御・採点境界・モデル障害のCPU/mock回帰テスト | [tests](../../tests) | **未達**: [#18](https://github.com/kmch4n/SkiSense/issues/18) |

## 運用

1. 仕様変更のPRでは、該当するIDについて「期待仕様」「実装」「テスト」「未検証」の状態を見直す。
2. 修正前に不一致をIssueへ記録し、受入条件を定義する。Issueに書かれた期待仕様を、実装済みの動作と混同しない。
3. 本表を自動で正しい状態に保つ仕組みは存在しない。CIのリンク・構文検査の成功だけで「仕様整合性を検証した」とは判断しない。
4. 新たなモデルの追加、スコア閾値の変更、対象選択方法の変更には、実行例と比較可能性の扱いを明記する。
