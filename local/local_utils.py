"""Cross-platform utilities for local SO-101 training, checkpointing, and dataset loading."""

import json
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEPLOY_LEROBOT_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)

SMOVLA_REQUIRED_FILES = ("training_state.pt",) + DEPLOY_LEROBOT_REQUIRED_FILES

DEPLOY_CUSTOM_REQUIRED_FILES = (
    "config.json",
    "model.pt",
)

CUSTOM_REQUIRED_FILES = ("training_state.pt",) + DEPLOY_CUSTOM_REQUIRED_FILES


def safe_sync():
    """Flush OS filesystem buffers on platforms that support os.sync()."""
    if hasattr(os, "sync"):
        os.sync()


def resolve_dataset_root(dataset_dir):
    """Locate a valid LeRobot dataset directory (containing meta/info.json) under dataset_dir."""
    base = Path(dataset_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(
            f"找不到資料集目錄：{base}\n請將 SO-101 資料集（含 data/、videos/、meta/）放入該資料夾。"
        )
    if not base.is_dir():
        raise NotADirectoryError(f"--dataset-dir 必須是資料夾而非檔案：{base}")

    if (base / "meta" / "info.json").is_file():
        return str(base)

    candidates = sorted(
        {
            p.parent.parent
            for p in base.rglob("meta/info.json")
            if p.is_file() and "__MACOSX" not in p.parts
        }
    )
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        found = ", ".join(str(c) for c in candidates)
        raise ValueError(
            f"在 {base} 下找到多個資料集：{found}\n請使用 --dataset-dir 指定確切的資料集目錄。"
        )

    raise FileNotFoundError(
        f"在 {base} 中找不到有效的 LeRobot 資料集（缺少 meta/info.json）。\n"
        "請確認已將資料集內容（data/、videos/、meta/）或 so101_pick_stylus/ 資料夾放入 ./dataset 中。"
    )


def verify_checkpoint_dir(
    ckpt_dir,
    required_files,
    check_processor_states=False,
    expected_arch=None,
    expected_type=None,
):
    """Verify required checkpoint files, optional processor states, and architecture/type match."""
    if not os.path.isdir(ckpt_dir):
        return False
    for fname in required_files:
        fpath = os.path.join(ckpt_dir, fname)
        if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
            return False

    if expected_arch is not None or expected_type is not None:
        try:
            with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if expected_arch is not None and cfg.get("arch") != expected_arch:
                return False
            if expected_type is not None and cfg.get("type") != expected_type:
                return False
        except Exception:
            return False

    if check_processor_states:
        for cfg_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            try:
                with open(os.path.join(ckpt_dir, cfg_name), "r", encoding="utf-8") as f:
                    proc_cfg = json.load(f)
                for step_cfg in proc_cfg.get("steps", []):
                    state_file = step_cfg.get("state_file")
                    if state_file:
                        s_path = os.path.join(ckpt_dir, state_file)
                        if not os.path.isfile(s_path) or os.path.getsize(s_path) == 0:
                            return False
            except Exception:
                return False
    return True


def recover_atomic_checkpoints(checkpoint_root):
    """Clean interrupted _tmp/_ckpt_staging dirs and restore _old backups after crashes."""
    if not os.path.isdir(checkpoint_root):
        return
    for name in os.listdir(checkpoint_root):
        full_path = os.path.join(checkpoint_root, name)
        if name == "_ckpt_staging" or (
            name.startswith("_") and (name.endswith("_tmp") or "_tmp_" in name)
        ):
            shutil.rmtree(full_path, ignore_errors=True)
    for name in sorted(os.listdir(checkpoint_root), reverse=True):
        if name.startswith("_") and (name.endswith("_old") or "_old_" in name):
            orig_name = name[1:].rsplit("_old", 1)[0]
            old_dir = os.path.join(checkpoint_root, name)
            orig_dir = os.path.join(checkpoint_root, orig_name)
            if not os.path.exists(orig_dir):
                os.rename(old_dir, orig_dir)
            else:
                shutil.rmtree(old_dir, ignore_errors=True)


def find_latest_checkpoint(checkpoint_root, required_files, check_processor_states=False, expected_arch=None):
    """Return (checkpoint_dir, training_state_dict) with the highest completed step."""
    best_dir, best_ckpt, best_step = None, None, -1
    if not os.path.isdir(checkpoint_root):
        return None, None
    for name in sorted(os.listdir(checkpoint_root)):
        if name.startswith("_"):
            continue
        cand_dir = os.path.join(checkpoint_root, name)
        if not verify_checkpoint_dir(
            cand_dir,
            required_files,
            check_processor_states=check_processor_states,
            expected_arch=expected_arch,
        ):
            continue
        try:
            ckpt = torch.load(
                os.path.join(cand_dir, "training_state.pt"),
                map_location="cpu",
                weights_only=False,
            )
            ckpt_step = int(ckpt.get("step", -1))
            if ckpt_step > best_step or (ckpt_step == best_step and name == "latest"):
                best_step = ckpt_step
                best_dir = cand_dir
                best_ckpt = ckpt
        except Exception as e:
            print(f"Warning: skipping corrupted checkpoint {cand_dir}: {e}")
    return best_dir, best_ckpt


def _rename_with_retry(src, dst, retries=6, delay=0.25):
    """Retry os.rename on transient Windows PermissionError (e.g., Defender / indexer locks)."""
    for attempt in range(retries):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def sync_dir_atomically(src_dir, dst_dir):
    """Atomically replace dst_dir with a complete copy of src_dir, rolling back on swap failure."""
    dst_path = Path(dst_dir).resolve()
    dst_str = str(dst_path)
    parent_dir = str(dst_path.parent)
    base_name = dst_path.name
    tmp_dir = os.path.join(parent_dir, f"_{base_name}_tmp")
    old_dir = os.path.join(parent_dir, f"_{base_name}_old")
    os.makedirs(parent_dir, exist_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(old_dir, ignore_errors=True)
    if os.path.exists(tmp_dir):
        tmp_dir = os.path.join(parent_dir, f"_{base_name}_tmp_{int(time.time() * 1000)}")
    if os.path.exists(old_dir):
        old_dir = os.path.join(parent_dir, f"_{base_name}_old_{int(time.time() * 1000)}")

    shutil.copytree(src_dir, tmp_dir, dirs_exist_ok=True)
    safe_sync()

    moved_old = False
    if os.path.exists(dst_str):
        _rename_with_retry(dst_str, old_dir)
        moved_old = True
    try:
        _rename_with_retry(tmp_dir, dst_str)
    except Exception:
        if moved_old and os.path.exists(old_dir) and not os.path.exists(dst_str):
            try:
                _rename_with_retry(old_dir, dst_str)
            except Exception:
                pass
        raise

    shutil.rmtree(old_dir, ignore_errors=True)
    safe_sync()


def build_training_state_payload(current_step, optimizer, scheduler, losses):
    """Snapshot optimizer, scheduler, losses, and RNG states for exact resume."""
    return {
        "step": current_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "losses": losses,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def restore_rng_state(rng_state):
    """Restore Python, NumPy, PyTorch CPU and CUDA RNG states from checkpoint."""
    if rng_state is None:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def build_trimmed_subset(dataset, ds_root, move_thresh=3.0, keep_lead_frames=10):
    """Trim leading idle frames per episode so the policy does not overfit stationary starts."""
    files = sorted(Path(ds_root).glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"在 {ds_root}/data/*/*.parquet 找不到任何 parquet 資料檔。")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sort_cols = [c for c in ("episode_index", "frame_index", "index") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    states = np.stack(df["observation.state"].values)
    eps = df["episode_index"].values
    indices = df["index"].values

    valid_indices = []
    for ep in np.unique(eps):
        m = eps == ep
        ep_states = states[m]
        ep_idx = indices[m]
        diffs = np.max(np.abs(ep_states - ep_states[0]), axis=1)
        moved = np.where(diffs > move_thresh)[0]
        start_offset = max(0, int(moved[0]) - keep_lead_frames) if len(moved) > 0 else 0
        valid_indices.extend(ep_idx[start_offset:].tolist())

    removed = len(df) - len(valid_indices)
    print(
        f"Trimmed idle prefix: {len(df)} -> {len(valid_indices)} frames "
        f"(removed {removed}, {removed / len(df) * 100:.1f}%)"
    )
    return torch.utils.data.Subset(dataset, valid_indices)


def save_loss_plot(losses, ylabel, title, out_path):
    """Save training loss curve to disk without blocking headless runs."""
    if not losses:
        return
    abs_out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    try:
        ax.plot(range(1, len(losses) + 1), losses, alpha=0.7)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if all(v > 0 for v in losses):
            ax.set_yscale("log")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(abs_out, dpi=150)
        print(f"Saved loss curve: {abs_out}")
    finally:
        plt.close(fig)
