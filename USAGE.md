# SO-101 腳本使用指南（USAGE）

本文件彙整 SO-101 機械手臂所有硬體檢測、訓練、離線測試與實機部署腳本範例。環境建置與目錄結構說明請參閱 [README.md](README.md)。

---

## 目錄
1. [相機編號掃描](#1-相機編號掃描)
2. [本機訓練指令（新資料集）](#2-本機訓練指令新資料集)
3. [本機離線 Dry-Run 測試](#3-本機離線-dry-run-測試不需連接實體手臂)
4. [SO-101 單任務 / 多任務實機部署](#4-so-101-單任務--多任務實機部署)
5. [多模型串接部署（`deploy_sequence.py`）](#5-多模型串接部署deploy_sequencepy)
6. [閉迴路 5 階段連續任務部署（YOLOE + SmolVLA）](#6-閉迴路-5-階段連續任務部署yoloe--smolvla)
7. [常用實機微調參數與輔助工具](#7-常用實機微調參數與輔助工具)

---

## 1. 相機編號掃描

在進行本機實機部署前，建議先確認兩支相機（`wrist` 與 `side_45`）對應的系統編號：
```bash
# 掃描並列出所有可用相機編號（不開啟預覽視窗）
uv run python -u tools/camera_check.py --scan-only

# 掃描並開啟即時預覽視窗（畫面左上角會顯示 ID，按 q 或 ESC 離開）
uv run python -u tools/camera_check.py
```

---

## 2. 本機訓練指令（新資料集）

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

## 3. 本機離線 Dry-Run 測試（不需連接實體手臂）

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

## 4. SO-101 單任務 / 多任務實機部署

`local/deploy_local.py` 預設 `--arch auto` 會自動從 Checkpoint 的 `config.json` 辨識模型架構（`smolvla` / `act` / `cnn` / `fc`），並自動對應資料集的雙相機鍵值（新資料集為 `--cam-wrist` 與 `--cam-side-45`，舊資料集仍相容 `--cam-top` 與 `--cam-front`）。

> 若現場相機名稱或編號不同，也可用通用參數覆寫，例如：`--cam wrist=0 side_45=2`。

### 4.1 SmolVLA（單任務：`so101_pick_cube` / `so101_pick_teabag`）
> SmolVLA 會讀取 `--task` 語言指令，請務必與資料集 `meta/tasks.parquet` 一致。

```bash
# Pick up the cube
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

# Pick up the teabag
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/so101_pick_teabag/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --task "Pick up the teabag" \
    --max-relative-target 5.0 \
    --display \
    --offline

# 額外 style_bs16_teabag Checkpoint 範例
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/so101_pick_style_bs16_teabag/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --task "Pick up the teabag" \
    --max-relative-target 5.0 \
    --display \
    --offline
```

### 4.2 SmolVLA（Multitask 多階段指令）
```bash
# 單一階段指令：Pick up the cube
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/multitask/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --tasks "Pick up the cube" \
    --stage-steps 800 \
    --max-relative-target 5.0 \
    --display \
    --offline

# 單一階段指令：put the object into the cup
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/multitask/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --tasks "put the object into the cup" \
    --stage-steps 800 \
    --max-relative-target 5.0 \
    --display \
    --offline

# 連續多階段指令：Pick up the cube -> put the object into the cup
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/smolvla/multitask/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --tasks "Pick up the cube" "put the object into the cup" \
    --stage-steps 800 800 \
    --max-relative-target 5.0 \
    --display \
    --offline
```

### 4.3 ACT（`so101_pick_teabag` / `so101_pick_cube` / `so101_put_into_cup` / `multitask`）
> ACT 僅使用視覺與關節狀態，不需要 `--task`（亦不需 `--offline`）。

```bash
uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/act/so101_pick_teabag/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --max-relative-target 5.0 \
    --display

uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/act/so101_pick_cube/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --max-relative-target 5.0 \
    --display

uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/act/so101_put_into_cup/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --max-relative-target 5.0 \
    --display

uv run python -u local/deploy_local.py \
    --checkpoint ./outputs/act/multitask/checkpoints/latest \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --max-relative-target 5.0 \
    --display
```

### 4.4 CNN（`so101_pick_cube` / `so101_pick_teabag`）
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

### 4.5 FC (MLP)（`so101_pick_cube` / `so101_pick_teabag`）
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

---

## 5. 多模型串接部署（`deploy_sequence.py`）

支援在單一連線會話中依固定步數依序切換多個模型（可混搭 `SmolVLA` / `ACT` / `CNN` / `FC`）：

```bash
# 混搭 SmolVLA + ACT 串接
uv run python -u local/deploy_sequence.py \
    --checkpoints \
        ./outputs/smolvla/so101_pick_cube/checkpoints/latest \
        ./outputs/act/so101_put_into_cup/checkpoints/latest \
        ./outputs/smolvla/so101_pick_teabag/checkpoints/latest \
        ./outputs/act/so101_put_into_cup/checkpoints/latest \
    --tasks "Pick up the cube" "" "Pick up the teabag" "" \
    --stage-steps 750 750 750 750 \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --display \
    --offline

# 純 SmolVLA（單任務 + multitask）串接
uv run python -u local/deploy_sequence.py \
    --checkpoints \
        ./outputs/smolvla/so101_pick_cube/checkpoints/latest \
        ./outputs/smolvla/multitask/checkpoints/latest \
        ./outputs/smolvla/so101_pick_teabag/checkpoints/latest \
        ./outputs/smolvla/multitask/checkpoints/latest \
    --tasks "Pick up the cube" "put the object into the cup" "Pick up the teabag" "put the object into the cup" \
    --stage-steps 750 750 750 750 \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --display \
    --offline

# 模型之間自動回 Home 並等待按 Enter 重置桌面物件
uv run python -u local/deploy_sequence.py \
    --checkpoints \
        ./outputs/smolvla/so101_pick_teabag/checkpoints/latest \
        ./outputs/smolvla/so101_pick_cube/checkpoints/latest \
    --tasks "Pick up the teabag" "Pick up the cube" \
    --stage-steps 750 750 \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --display \
    --offline \
    --return-home-between --wait-enter-between
```

---

## 6. 閉迴路 5 階段連續任務部署（YOLOE + SmolVLA）

整合 [`local/smolvla_pool.py`](local/smolvla_pool.py) 與 [`local/yolo_monitor.py`](local/yolo_monitor.py)，由 [`local/deploy_continuous_task.py`](local/deploy_continuous_task.py) 自動執行完整 5 階段飲品調製流程，並支援視覺抓取確認、不透明杯放置驗證、中途掉落自動回退與逾時張爪歸位重試：
1. **Stage 1**：`Pick Cube`（抓取方塊）
2. **Stage 2**：`Put Cube into Cup`（將方塊放入杯中，若中途掉落自動回退至 Stage 1）
3. **Stage 3**：`Pick Teabag`（抓取茶包）
4. **Stage 4**：`Put Teabag into Cup`（將茶包放入杯中，若中途掉落自動回退至 Stage 3）
5. **Stage 5**：`Shake Cup`（搖晃杯子）

### 6.1 YOLOE 視覺提示校準（不需訓練資料集）
本專案的 YOLOE 採用 **Visual Prompting（視覺提示）**，只需提供 **1 張參考樣本圖 + 框選 3 個目標物件（`tea bag`、`cube`、`cup`）**，即可在主迴圈中即時偵測：

1. **安裝依賴**（若尚未安裝 `ultralytics`；權重預設為 `yoloe-11l-seg.pt`，首次執行會自動下載）：
   ```bash
   uv add ultralytics
   ```
2. **建立樣本參考框**（二選一，會產生 `local/yoloe_ref.jpg` 與 `local/yoloe_prompt_config.json`）：
   ```bash
   # 方式 A（推薦）：使用 side_45 相機現場拍攝樣本
   # 將 tea bag / cube / cup 擺好並移開手臂 -> 按 's' 擷取畫面 -> 依序拖曳框選 3 個物件並按 ENTER/SPACE 確認
   uv run python -u local/yolo_monitor.py --calibrate-ref --cam-side-45 2

   # 方式 B：使用現有樣本照片（例如從資料集 side_45 影片擷取之第 0 幀）
   uv run python -u local/yolo_monitor.py --calibrate-ref --image <樣本圖片路徑>
   ```
3. **驗證偵測效果**（按 `b` 可更新基準框，按 `q` 離開；若有漏抓可調低 `--yolo-conf`，預設 `0.25`）：
   ```bash
   uv run python -u local/yolo_monitor.py --display --cam-side-45 2
   ```
   > 若未來改用自行 Fine-tune 的標準 YOLO 權重（`.pt`），可在 `deploy_continuous_task.py` 加上 `--yolo-mode standard --yolo-weights <best.pt>`。

### 6.2 執行 5 階段連續部署與驗證
```bash
# 1. 預檢 4 個 SmolVLA Checkpoint 完整性並量測 5 階段切換延遲
uv run python -u tools/verify_checkpoints.py --benchmark
# （若僅想在無 checkpoint 機器上驗證流程，可改用 --mock）
uv run python -u tools/verify_checkpoints.py --mock

# 2. 離線模擬測試完整狀態機（不需連接手臂或模型權重，並模擬超時與掉落恢復）
uv run python -u local/deploy_continuous_task.py --dry-run --mock-models --simulate-failures

# 3. 實機閉迴路 5 階段連續部署
uv run python -u local/deploy_continuous_task.py \
    --port /dev/ttyACM0 \
    --robot-id my_follower \
    --cam-wrist 0 \
    --cam-side-45 2 \
    --preload-mode auto \
    --display \
    --offline
```
> - `--preload-mode auto` 會在 VRAM `>= 8 GB`（如 RTX 3060 12GB / AMD AI Max+ 395）時自動選用 `gpu_full` 零拷貝切換；低 VRAM 裝置可選 `gpu_experts` 或 `cpu_pinned`。
> - 啟用 `--display` 時，HUD 視窗支援熱鍵操作：按 `n` 跳至下一階段、`r` 張爪掉落後重設當前階段、`q` 結束部署。

---

## 7. 常用實機微調參數與輔助工具

- **歸位工具**：`uv run python -u tools/go_home.py --port /dev/ttyACM0 --robot-id my_follower` 可平滑將手臂移動回 Home 姿態。
- `--action-ema 0.5`（範圍 `(0, 1]`）：對連續動作進行指數移動平均平滑，可減少機械臂抖動。
- `--max-relative-target 5.0`：限制單步關節角度最大變化量，提升實機安全性。
- `--display`：於部署時顯示即時雙相機預覽視窗（`deploy_local.py` 中按 `r` 會先張開夾爪再回 Home 重試，按 `q` 停止）。
