# ポーズ推定バックエンド

[English](pose_backends.md) | **日本語**

SkiSense は姿勢推定エンジンを 2 種類から選べる。`yolo11` は移行前のエンジンで
あり、非推奨ではなく正式にサポートを継続している。したがって旧来の 2D 動作へは
常にフラグ 1 つで戻せる。

**実行ごとの切り替え — CLI フラグ**（最優先）:

```bash
python run.py --pose-backend yolo11 video.mp4     # 旧来の 2D エンジン
python run.py --pose-backend sam3d  video.mp4     # 3D エンジン
python run.py --pose-backend yolo11 skier.jpg --image
```

**恒久的な既定値 — `.env`:**

```bash
SKISENSE_POSE_BACKEND=sam3d    # 既定: SAM 3D Body（3D・高精度）
SKISENSE_POSE_BACKEND=yolo11   # YOLO11-Pose（2D・軽量・高速）
```

優先順位は `--pose-backend` → `SKISENSE_POSE_BACKEND` → `sam3d` である。
選択されたエンジンは起動時バナーの `- Pose:` 行に出力されるため、長時間の
処理を始める前にどちらが有効か確認できる。

人物検出（YOLOv8x）・トラッキング（Deep SORT）・ズーム・スコアリングの
パイプラインは共通で、姿勢推定部だけが差し替わる。各バックエンドは自身の
トポロジ（`pose_topology.py`）を宣言し、`analyze_ski_pose` と描画ヘルパーが
それに自動追従する。`--fast` / `--high` / `--target-mode` はどちらのエンジン
でも動作する。

## 比較表

| 項目 | SAM 3D Body | YOLO11-Pose |
|---|---|---|
| バックエンド名 | `sam3d`（既定） | `yolo11`（旧来モード） |
| トポロジ | MHR-21（体 + 足先 + 手首） | COCO-17 |
| 次元 | 3D（カメラ座標）+ 2D 投影 | 2D 画像座標のみ |
| 関節角度 | 膝・股関節・足首を 3D で算出（視点不変） | 膝・股関節を 2D で算出 |
| 足首角度 | ◎ 取得可（足先ランドマークあり） | ✗ N/A（COCO-17 に足先なし） |
| 肩の傾き | 2D 画像平面 | 2D 画像平面 |
| デバイス | **CUDA 必須** | CPU / MPS / CUDA |
| 速度（目安） | ~1–2 s/frame（RTX 4060） | 数 ms/ROI |
| VRAM（目安） | ~3.4 GB（FP16, body-only） | ~2–3 GB |
| 重み入手 | HuggingFace ゲート（要申請 + `hf auth login`） | Ultralytics 自動 DL |
| ライセンス | SAM License（Apache 2.0 相当・商用可） | AGPL-3.0（Ultralytics） |
| メッシュ出力 | ◎ あり（`pred_vertices`） | ✗ なし |

## SAM 3D Body（`sam3d`）

Meta が 2025-11-19 に公開した単一画像 3D 人体メッシュ復元モデル。MHR
（Momentum Human Rig）パラメトリックモデルで、視点に依存しない真の関節角度が
得られる。SkiSense は MHR70 キーポイントの先頭 63（体・足先・手首）を使用する。

- **強み**: 斜め視点でも膝・股関節・足首角度が歪まない。内脚の遮蔽に強い。
  足首の前圧（knee→ankle→toe）まで評価でき、人体メッシュも取得できる。
- **弱み**: CUDA 必須。1 フレーム数秒と重く、動画はオフライン処理前提。
  重みがゲート公開のため初回に HuggingFace のアクセス申請が要る。
- **設定**:
  - `SKISENSE_SAM3D_HF_REPO`: `facebook/sam-3d-body-dinov3`（既定, 840M）/
    `facebook/sam-3d-body-vith`（631M, 軽量・低 VRAM）
  - `SKISENSE_SAM3D_USE_HAND_REFINE`: 手指デコーダ。スキー採点では不要のため
    既定 `false`（VRAM・レイテンシをほぼ半減）
- **セットアップ**: [`../notes/sam3d_setup.md`](../notes/sam3d_setup.md) を参照
  （detectron2 / MoGe は不要）。

## YOLO11-Pose（`yolo11`）

Ultralytics の 2D 姿勢推定モデルであり、SAM 3D Body 移行前に SkiSense が使って
いたエンジンでもある。COCO-17 キーポイントを ROI 単位で推定する。旧パイプライン
の結果を再現できるようにし、また CUDA 非搭載機でも動かせるようにするため、
非推奨扱いにせず意図的に維持している。

- **強み**: CPU でも動く。1 フレーム数 ms と高速で、リアルタイム寄りの確認や
  CUDA 非搭載機での利用に向く。重みは自動ダウンロード。
- **弱み**: 2D のため斜め視点で関節角度に投影歪みが出る。COCO-17 に足先
  ランドマークがないため**足首角度は N/A**（スコアから除外）。
- **設定**:
  - `SKISENSE_YOLO_POSE_MODEL`: `yolo11x-pose.pt`（既定）他 n/s/m/l/x
  - `SKISENSE_YOLO_POSE_CONFIDENCE`: キーポイント信頼度しきい値（既定 0.25）
  - `SKISENSE_CLAHE_ENABLED`: ROI への CLAHE 適用（既定 false）
  - `SKISENSE_FLIP_TTA_ENABLED`: 水平反転 TTA（既定 false, 推論コスト 2 倍）

## 使い分けの指針

- **精密なフォーム分析・本番の可視化** → `sam3d`。視点不変の 3D 角度と足首評価、
  メッシュが活きる。CUDA があり処理時間を許容できる場合。
- **素早い確認・CUDA 非搭載機・大量バッチの一次スクリーニング** → `yolo11`。
- **移行前の出力を再現したい場合** → `yolo11`。SAM 3D Body 移行前と同一の
  エンジン・同一の COCO-17 トポロジである。

## バックエンドをまたいだスコアは比較できない

バックエンドを切り替えると、精度だけでなく測定値そのものが変わる。したがって
`sam3d` のスコアと `yolo11` のスコアを直接比べてはならない。

- 膝・股関節の角度は、`sam3d` ではカメラ空間の 3D ベクトルから、`yolo11` では
  画像平面に投影された座標から算出する。同じ姿勢でも斜め視点では度数が異なる。
- 総合スコアは測定できた関節の平均である。`yolo11` は左右の足首角度を測定できない
  ため、7 項目ではなく 5 項目の平均になる。

シーズンを通したクリップ群を見比べるときなど、比較を行う場面ではバックエンドを
固定すること。

## 既知の制約

- `yolo11` は COCO-17 トポロジのため足首角度を出せない（info panel は `N/A`）。
- `sam3d` は CUDA 必須。`SKISENSE_DEVICE` が CUDA に解決されない環境で `sam3d` を
  選ぶと、バックエンド構築時に `RuntimeError` を送出する。エラーメッセージは
  `--pose-backend yolo11` への切り替えを案内する。
- `sam3d` は関節ごとの信頼度を公開しないため、使用する関節の visibility は一律
  `1.0` で埋めている。その結果 `SKISENSE_POSE_VISIBILITY_THRESHOLD` および
  `SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS` は `sam3d` では効果を持たず、遮蔽された
  関節もモデルの推定値のまま採点される。`yolo11` では両設定とも有効である。
