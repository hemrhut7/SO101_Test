# SO-101 機械手臂訓練與部署專案

本專案用於在 **SO-101** 雙相機機械手臂上訓練與部署 **SmolVLA**、**ACT**、**CNN** 與 **FC (MLP)** 策略，並統一使用 **`uv`** 管理本機環境，與 Colab 環境共同鎖定 **`lerobot==0.7.0`**（Python 3.12）。

---

## 專案目錄結構

```text
SO101_Test/
├── .python-version                      # 鎖定 Python 3.12（uv 會自動下載並使用）
├── pyproject.toml                       # 專案設定與頂層相依套件清單
├── uv.lock                              # 完整相依套件鎖定檔（由 uv 自動產生與維護）
├── colab/                               # Google Colab GPU 訓練 Notebook（皆鎖定 lerobot==0.7.0）
│   ├── smolVLA_train.ipynb              # 原始範例（so101_pick_stylus）
│   ├── smolVLA_train_pick_cube.ipynb    # SmolVLA 微調（so101_pick_cube）
│   ├── smolVLA_train_pick_teabag.ipynb  # SmolVLA 微調（so101_pick_teabag）
│   ├── act_train.ipynb
│   ├── cnn_train.ipynb
│   └── fc_train.ipynb
├── local/                               # 本機訓練與 SO-101 實機部署腳本
│   ├── train_local.py
│   ├── deploy_local.py
│   ├── local_policies.py
│   └── local_utils.py
├── tools/                               # 硬體檢測與輔助工具
│   ├── camera_check.py                  # 自動掃描可用相機編號並開啟即時預覽
│   └── go_home.py
├── datasets/                            # LeRobot v3.0 資料集
│   ├── so101_pick_cube/                 # 相機：wrist, side_45 | task: "Pick up the cube"
│   └── so101_pick_teabag/               # 相機：wrist, side_45 | task: "Pick up the teabag"
└── outputs/                             # 訓練輸出目錄（outputs/<arch>/<dataset_name>/checkpoints）
```

---

## 使用 `uv` 管理環境（新手快速指南）

> **什麼是 `pyproject.toml`、`uv.lock` 與 `.python-version`？**
> - `.python-version`：指定專案使用的 Python 版本（`3.12`）。
> - `pyproject.toml`：記錄專案需要哪些頂層套件（例如 `lerobot[smolvla,dataset,feetech]==0.7.0`）。
> - `uv.lock`：記錄**所有直接與間接相依套件**的精確版本與雜湊值（包含 `torch`、`transformers`、`feetech-servo-sdk` 等），確保在不同電腦或重新安裝時，環境 100% 一致。

### 1. 初次建置 / 同步環境
只要專案目錄內有 `.python-version`、`pyproject.toml` 與 `uv.lock`，執行一次即可自動建好 `.venv` 並裝好所有套件：
```bash
uv sync
```

### 2. 執行腳本（不需手動 activate `.venv`）
使用 `uv run` 會自動在專案的 `.venv` 內執行程式：
```bash
uv run python -u local/deploy_local.py --help
```

### 3. 新增或移除套件
```bash
# 新增套件（會自動更新 pyproject.toml 與 uv.lock）
uv add <套件名稱>

# 移除套件
uv remove <套件名稱>
```
> **注意**：請勿直接編輯 `uv.lock`，它由 `uv sync` / `uv add` / `uv remove` 自動維護；也建議將 `pyproject.toml`、`.python-version` 與 `uv.lock` 一併納入 Git 版本控制。

---

## 相機編號掃描

在進行本機實機部署前，建議先確認兩支相機（`wrist` 與 `side_45`）對應的系統編號：
```bash
# 掃描並列出所有可用相機編號（不開啟預覽視窗）
uv run python -u tools/camera_check.py --scan-only

# 掃描並開啟即時預覽視窗（畫面左上角會顯示 ID，按 q 或 ESC 離開）
uv run python -u tools/camera_check.py
```

---

## 本機訓練指令（新資料集）

`local/train_local.py` 會依 `--dataset` 自動將輸出隔離存放在 `./outputs/<arch>/<dataset_name>/checkpoints`，避免不同任務互相覆蓋：

```bash
# 1. SmolVLA：訓練 so101_pick_cube
uv run python -u local/train_local.py --arch smolvla --dataset so101_pick_cube --batch-size 16 --steps 25000

# 2. SmolVLA：訓練 so101_pick_teabag
uv run python -u local/train_local.py --arch smolvla --dataset so101_pick_teabag --batch-size 16 --steps 25000

# 3. ACT：訓練 so101_pick_teabag / so101_pick_cube
uv run python -u local/train_local.py --arch act --dataset so101_pick_teabag --batch-size 16 --steps 20000

# 4. CNN：訓練 so101_pick_cube
uv run python -u local/train_local.py --arch cnn --dataset so101_pick_cube --batch-size 32 --steps 15000

# 5. FC (MLP)：訓練 so101_pick_cube（可選 --fc-no-images 僅用關節狀態）
uv run python -u local/train_local.py --arch fc --dataset so101_pick_cube --batch-size 32 --steps 15000
```

---

## 本機離線 Dry-Run 測試（不需連接實體手臂）

```bash
# 測試 SmolVLA（so101_pick_cube）
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/so101_pick_cube/checkpoints/latest \
    --dataset-dir ./datasets/so101_pick_cube \
    --task "Pick up the cube" \
    --dry-run --offline

# 測試 ACT（so101_pick_teabag）
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/act/so101_pick_teabag/checkpoints/latest \
    --dataset-dir ./datasets/so101_pick_teabag \
    --dry-run
```

---

## SO-101 實機部署（不同模型與新資料集範例）

`local/deploy_local.py` 預設 `--arch auto` 會自動從 Checkpoint 的 `config.json` 辨識模型架構（`smolvla` / `act` / `cnn` / `fc`），並自動對應資料集的雙相機鍵值（新資料集為 `--cam-wrist` 與 `--cam-side-45`，舊資料集仍相容 `--cam-top` 與 `--cam-front`）。

> 若現場相機名稱或編號不同，也可用通用參數覆寫，例如：`--cam wrist=0 side_45=2`。

### 1. SmolVLA（`so101_pick_cube`）
> SmolVLA 會讀取 `--task` 語言指令，請務必與資料集 `meta/tasks.parquet` 一致。
```bash
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/so101_pick_cube/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --task "Pick up the cube" \
    --n-action-steps 35 \
    --max-relative-target 5.0 \
    --display \
    --offline
```

### 2. SmolVLA（`so101_pick_teabag`）
```bash
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/so101_pick_teabag/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --task "Pick up the teabag" \
    --n-action-steps 35 \
    --max-relative-target 5.0 \
    --display \
    --offline
```

### 3. ACT（`so101_pick_teabag` / `so101_pick_cube`）
> ACT 僅使用視覺與關節狀態，不需要 `--task`（亦不需 `--offline`）。
```bash
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/act/so101_pick_teabag/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --n-action-steps 35 \
    --max-relative-target 5.0 \
    --display
```

### 4. CNN（`so101_pick_cube` / `so101_pick_teabag`）
> CNN 預設 `--n-action-steps` 為 `15`。
```bash
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/cnn/so101_pick_cube/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --n-action-steps 15 \
    --max-relative-target 5.0 \
    --display
```

### 5. FC (MLP)（`so101_pick_cube` / `so101_pick_teabag`）
```bash
# 含雙相機影像的 FC 模型
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/fc/so101_pick_cube/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --n-action-steps 15 \
    --max-relative-target 5.0

# 若訓練時有加 --fc-no-images（純關節狀態模型），部署時需加上 --no-cameras
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/fc/so101_pick_cube/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --no-cameras \
    --n-action-steps 15 \
    --max-relative-target 5.0
```

### 常用實機微調參數
- `--action-ema 0.5`（範圍 `(0, 1]`）：對連續動作進行指數移動平均平滑，可減少機械臂抖動。
- `--max-relative-target 5.0`：限制單步關節角度最大變化量，提升實機安全性。
- `--display`：於部署時顯示即時雙相機預覽視窗（按 `q` 停止）。
