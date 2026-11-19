# SO-101 機械手臂訓練與部署專案

本專案用於在 **SO-101** 雙相機機械手臂（`observation.images.top`、`observation.images.front`）上訓練與部署 **SmolVLA**、**ACT**、**CNN** 與 **FC (MLP)** 策略，並統一使用 **`uv`** 管理本機環境，與 Colab 環境共同鎖定 **`lerobot==0.7.0`**（Python 3.12）。

---

## 專案目錄結構

```text
SO101_Test/
├── .python-version       # 鎖定 Python 3.12（uv 會自動下載並使用）
├── pyproject.toml        # 專案設定與頂層相依套件清單
├── uv.lock               # 完整相依套件鎖定檔（由 uv 自動產生與維護）
├── colab/                # Google Colab GPU 訓練 Notebook（皆鎖定 lerobot==0.7.0）
│   ├── smoVLA_train.ipynb
│   ├── act_train.ipynb
│   ├── cnn_train.ipynb
│   └── fc_train.ipynb
├── local/                # 本機訓練與 SO-101 實機部署腳本
│   ├── train_local.py
│   ├── deploy_local.py
│   ├── local_policies.py
│   └── local_utils.py
├── datasets/             # LeRobot v3.0 資料集（例如 so101_pick_stylus）
└── outputs/              # 訓練輸出目錄（outputs/<arch>/checkpoints）
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

## 常用訓練與部署指令

### 本機離線 Dry-Run 測試（不需連接實體手臂）
```bash
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/checkpoints/latest \
    --dataset-dir ./datasets/so101_pick_stylus \
    --dry-run --offline
```

### SO-101 實機部署
```bash
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_so101 \
    --cam-top 0 \
    --cam-front 1 \
    --task "Pick up the stylus from the stand" \
    --n-action-steps 35 \
    --max-relative-target 5.0 \
    --offline
```
- 若控制仍有些微震動，可加上 `--action-ema 0.5`（範圍 `(0, 1]`）對連續動作進行指數移動平均平滑。

### 本機訓練（以 SmolVLA 為例）
```bash
uv run python -u local/train_local.py \
    --arch smolvla \
    --dataset-dir ./datasets/so101_pick_stylus \
    --steps 15000
```
