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


def build_trimmed_subset(
    dataset,
    ds_root,
    move_thresh=3.0,
    keep_lead_frames=10,
    keep_tail_frames=10,
    trim_head=True,
    trim_tail=True,
    drop_longest_n=10,
    exclude_episodes=None,
):
    """Filter out longest/excluded episodes and trim idle head/tail frames per episode."""
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
    timestamps = df["timestamp"].values if "timestamp" in df.columns else None

    unique_eps = np.unique(eps)
    ep_masks = {int(ep): (eps == ep) for ep in unique_eps}

    excluded_set = {int(e) for e in (exclude_episodes or ())}
    remaining_eps = [int(ep) for ep in unique_eps if int(ep) not in excluded_set]

    longest_dropped = []
    if drop_longest_n > 0:
        if len(remaining_eps) <= drop_longest_n:
            raise ValueError(
                f"可用 episode 數 ({len(remaining_eps)}) 不足，無法排除最長的 {drop_longest_n} 筆。"
            )
        ep_durations = []
        for ep in remaining_eps:
            m = ep_masks[ep]
            n_frames = int(np.sum(m))
            if timestamps is not None and n_frames > 1:
                t_start = float(np.asarray(timestamps[m][0]).item())
                t_end = float(np.asarray(timestamps[m][-1]).item())
                dur = t_end - t_start
            else:
                dur = float(n_frames)
            ep_durations.append((dur, n_frames, ep))
        ep_durations.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        for dur, n_frames, ep in ep_durations[:drop_longest_n]:
            excluded_set.add(ep)
            longest_dropped.append((ep, n_frames, dur))

    valid_indices = []
    dropped_ep_frames = 0
    trimmed_head_frames = 0
    trimmed_tail_frames = 0

    for ep in unique_eps:
        ep_int = int(ep)
        m = ep_masks[ep_int]
        ep_idx = indices[m]
        if ep_int in excluded_set:
            dropped_ep_frames += len(ep_idx)
            continue

        ep_states = states[m]
        n_ep = len(ep_states)

        if trim_head:
            head_diffs = np.max(np.abs(ep_states - ep_states[0]), axis=1)
            head_moved = np.where(head_diffs > move_thresh)[0]
            start_offset = max(0, int(head_moved[0]) - keep_lead_frames) if len(head_moved) > 0 else 0
        else:
            start_offset = 0

        if trim_tail:
            tail_diffs = np.max(np.abs(ep_states - ep_states[-1]), axis=1)
            tail_moved = np.where(tail_diffs > move_thresh)[0]
            end_offset = (
                min(n_ep, int(tail_moved[-1]) + 1 + keep_tail_frames)
                if len(tail_moved) > 0
                else n_ep
            )
        else:
            end_offset = n_ep

        if end_offset <= start_offset:
            start_offset = 0
            end_offset = n_ep

        trimmed_head_frames += start_offset
        trimmed_tail_frames += n_ep - end_offset
        valid_indices.extend(ep_idx[start_offset:end_offset].tolist())

    if longest_dropped:
        dropped_desc = ", ".join(
            f"ep{ep}({n}f/{dur:.1f}s)" if timestamps is not None else f"ep{ep}({n}f)"
            for ep, n, dur in longest_dropped
        )
        print(f"Excluded longest {len(longest_dropped)} episodes: {dropped_desc}")
    if len(excluded_set) > len(longest_dropped):
        manual_eps = sorted(excluded_set - {ep for ep, _, _ in longest_dropped})
        print(f"Excluded manual episodes: {manual_eps}")

    removed = len(df) - len(valid_indices)
    print(
        f"Dataset filter summary: {len(df)} -> {len(valid_indices)} frames "
        f"(kept {len(valid_indices) / len(df) * 100:.1f}% | "
        f"dropped_eps={dropped_ep_frames}, trimmed_head={trimmed_head_frames}, "
        f"trimmed_tail={trimmed_tail_frames}, total_removed={removed})"
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
