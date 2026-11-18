# 現行動作仕様（as-is・提案中）

- **基準:** 2026-11-18 / `main` `0fa4216f27bb907e5f99272831faff56ba42c2d0`
- **状態:** Proposed。静的コード監査に基づく記述。CUDA/実動画での受入テストは未実施。
- **一次的な実装参照:** `run.py`、`compare.py`、`src/skisense/main.py`、`image_processor.py`、`backends/`、`config.py`

## 目的と非目的

映像中の人物を検出・追跡し、骨格推定と関節角度の評価結果を描画する。**スキー競技の審判点数を予測・再現する仕組みではない。** 現行の0〜100点はヒューリスティックな参考指標である。

## CLI・設定

| 指定 | 現行動作 | 制約・既知の問題 |
| --- | --- | --- |
| `python run.py video.mp4` | 動画処理 | 通常は `input/` 以下から読込 |
| `python run.py skier.jpg --image` | 静止画処理 | 最大面積の人物を選ぶ |
| `--pose-backend yolo26` | 既定の2D COCO-17モデル | 足先座標なし、足首角度N/A |
| `--pose-backend yolo11` | 従来の2D COCO-17モデル | 足先座標なし |
| `--pose-backend sam3d` | MHR 3D＋2D投影 | CUDAとゲート公開重み・外部ソースが必要 |
| `--fast` | バックエンドの全画面推論 | YOLO系は全画面で複数人を検出。SAM3Dは検出器なし・単一の画像全体bboxにフォールバック（#16）。`longest`＋ズーム有効時だけ事前走査あり |
| `--target-mode longest` | 最長出演trackを事前選択する設定 | 既定。ただし現在はズーム有効時だけ事前走査する。無効時は選択仕様が変わる（#15）。選定失敗時はlargestへフォールバック |
| `--target-mode largest` | 通常＋ズーム有効時は既存追跡IDを優先し、選び直す時に最大のconfirmed trackを使う | 高速モードは毎フレーム最大の姿勢検出bboxを使う。通常＋ズーム無効時は最初の姿勢結果になり、設定は保証されない（#15） |
| `--high` | 実質何も変更しない | **未実装、Issue #4** |
| `SKISENSE_ZOOM_ENABLED=false` | ズームを無効化 | **姿勢描画欠落の既知バグ、Issue #15** |

CLIの`--pose-backend`は`.env`より優先され、未指定時は`yolo26`。画像モードは動画専用フラグと併用不可。`--high`と`--fast`の併用も不可。

### 信頼度とズーム設定の契約

- `SKISENSE_YOLO_CONFIDENCE`（既定0.3）：通常動画のYOLOv8x人物検出に渡す信頼度しきい値。YOLO-Pose側の検出と関節可視性のしきい値とは別。
- `SKISENSE_YOLO_POSE_CONFIDENCE`（既定0.25）：YOLO11/YOLO26共通のUltralytics姿勢推定モデルに`conf=`として渡す**人物／インスタンス検出信頼度**のしきい値。ROI・全画面の両推論経路で使用する。個別関節をフィルタする設定ではない。
- `SKISENSE_POSE_VISIBILITY_THRESHOLD`（既定0.5）と`SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS`（既定0.3）：YOLO-Poseの`keypoints.conf`から設定した各ランドマークのvisibilityを採点・描画時に判定するしきい値。SAM3Dは対象関節のvisibilityに原則1.0を代入するため同じ意味を持たない。
- `SKISENSE_ZOOM_TARGET_AREA_RATIO`と`SKISENSE_ZOOM_MAX_SCALE`：現在の動的ズーム倍率の計算に使用する。
- `SKISENSE_ZOOM_SCALE`、`SKISENSE_ZOOM_SMOOTHING`、`SKISENSE_ZOOM_PADDING`：設定として受理・保持するが、現行の動的ズームや対象ロスト時の倍率決定には使用しない。値を変更しても現行処理の映像には反映されない（`zoom_tracker.py`を参照）。

## 映像処理

**通常動画:** 選択バックエンドを初期化→動画を開きFPS・サイズを検証→（ズーム有効かつlongestなら）主対象の事前追跡→YOLOv8x検出→各bboxについて姿勢推定→対象を対応付け→ズーム・骨格・パネル合成→MP4書込。ズーム無効時は対象のbbox/骨格描画が欠落する既知バグがある（#15）。

**高速動画:** 毎フレームのYOLOv8x/Deep SORTとROI推論を省き、バックエンドの全画面推論を使う。YOLO系は全画面で人物と骨格を検出する。SAM3Dはbbox未指定・検出器無効のため単一画像全体bboxを入力とする（#16）。`longest`＋ズーム有効時はYOLOv8x＋Deep SORTの事前走査を行う。ズーム無効でも高速モードの骨格・bbox描画は行うが、longestの事前選択は省略される（#15）。

**SAM3D高速モードの制約（Issue #16）:** `Sam3dBackend`は`human_detector=None`で初期化され、全画面モードはbboxを渡さない。Metaの現行実装は検出器とbboxがともにない場合、画像全体の単一bboxを使用する。複数人を検出・識別できるものとして扱わない。実映像検証と修正が必要。

## 出力

- 通常動画：`output/YYYYMMDD_HHMMSS/video_pose.mp4`、入力コピー、最高得点フレームの`best_shot.jpg`（スコアが正の場合のみ。0点のフレームは保存されない）。出力ライター初期化の成否は未検証（#17）。
- 高速動画：`video_pose_fast.mp4`。
- 静止画：`image_pose.jpg`および入力コピー。
- モデル比較：`comparison_YYYYMMDD_HHMMSS/`に`index.html`、比較画像、`results.json`。

**既知の問題（Issue #17）:** 動画出力の`VideoWriter.isOpened()`確認は未実装。出力の成功を保証しない。

## 比較処理

`compare.py`は同一サンプルフレームの最大YOLOv8x bboxを選択し、`sam3d`、`yolo11`、`yolo26`、`rtmw3d`の出力を並べる。`rtmw3d`の3D座標系は未検証のため通常処理では選択不可。モデル間の点数比較は行わない。個別モデルの失敗はJSONに`error`で残す。

## 実装変更時の手順

Issue・期待動作の明文化→spec変更→決定的な受入テスト→実装→実モデル検証→PRレビューの順とする。実測していないことを「検証済み」と記載しない。
