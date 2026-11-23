"""Pre-flight validator and switching benchmark for cross-PC SmolVLA checkpoints.

Run this on the target deployment computer (RTX 3060 12GB / AMD AI Max+ 395) after
copying the 4 trained SmolVLA checkpoints, or run with ``--mock`` on a development
machine to verify the benchmark harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DIR = PROJECT_ROOT / "local"
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from local_utils import (  # noqa: E402
    DEPLOY_LEROBOT_REQUIRED_FILES,
    is_legacy_bf16_smolvla_checkpoint,
    verify_checkpoint_dir,
)
from smolvla_pool import SmolVLAPolicyPool, VALID_PRELOAD_MODES  # noqa: E402


def _hash_vlm_tensors(safetensors_path: Path) -> tuple[str, int, int]:
    from safetensors.torch import load_file

    sd = load_file(str(safetensors_path), device="cpu")
    h = hashlib.sha256()
    vlm_count = 0
    expert_count = 0
    for key in sorted(sd.keys()):
        if key.startswith("model.vlm_with_expert.vlm."):
            vlm_count += 1
            h.update(key.encode("utf-8"))
            h.update(sd[key].contiguous().view(torch.uint8).numpy().tobytes())
        else:
            expert_count += 1
    return h.hexdigest()[:16], vlm_count, expert_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 4 SmolVLA checkpoints and benchmark stage switching.")
    parser.add_argument("--ckpt-pick-cube", default=str(PROJECT_ROOT / "outputs/smolvla/so101_pick_cube/checkpoints/latest"))
    parser.add_argument("--ckpt-put-cup", default=str(PROJECT_ROOT / "outputs/smolvla/so101_put_into_cup/checkpoints/latest"))
    parser.add_argument("--ckpt-pick-teabag", default=str(PROJECT_ROOT / "outputs/smolvla/so101_pick_teabag/checkpoints/latest"))
    parser.add_argument("--ckpt-shake-cup", default=str(PROJECT_ROOT / "outputs/smolvla/so101_shake_cup/checkpoints/latest"))
    parser.add_argument("--preload-mode", default="auto", choices=list(VALID_PRELOAD_MODES))
    parser.add_argument("--device", default=None)
    parser.add_argument("--benchmark", action="store_true", help="Load pool and benchmark 5-stage switching latency.")
    parser.add_argument("--mock", action="store_true", help="Run switching benchmark with mock models (no checkpoints needed).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_map = {
        "pick_cube": args.ckpt_pick_cube,
        "put_cup": args.ckpt_put_cup,
        "pick_teabag": args.ckpt_pick_teabag,
        "shake_cup": args.ckpt_shake_cup,
    }

    print("=== Hardware Environment ===")
    print(f"PyTorch: {torch.__version__} | Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | Total VRAM: {props.total_memory / (1024 ** 3):.2f} GB")

    if not args.mock:
        print("\n=== Checkpoint Integrity & Backbone Compatibility ===")
        vlm_hashes: set[str] = set()
        cam_configs: set[tuple[str, ...]] = set()
        for key, raw_path in ckpt_map.items():
            ckpt_dir = Path(raw_path).expanduser().resolve()
            ok = verify_checkpoint_dir(
                str(ckpt_dir),
                DEPLOY_LEROBOT_REQUIRED_FILES,
                check_processor_states=True,
                expected_type="smolvla",
            )
            if not ok:
                raise FileNotFoundError(f"[{key}] Checkpoint 缺少必要檔案或驗證失敗：{ckpt_dir}")
            legacy_bf16 = is_legacy_bf16_smolvla_checkpoint(str(ckpt_dir))
            with open(ckpt_dir / "config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cams = tuple(sorted(k for k in cfg.get("input_features", {}) if k.startswith("observation.images.")))
            cam_configs.add(cams)
            vlm_hash, n_vlm, n_exp = _hash_vlm_tensors(ckpt_dir / "model.safetensors")
            vlm_hashes.add(vlm_hash)
            print(
                f"  [OK] {key:12s} | cameras={list(cams)} | vlm_hash={vlm_hash} ({n_vlm} tensors) "
                f"| expert={n_exp} tensors | legacy_bf16={legacy_bf16}"
            )
            if legacy_bf16:
                print(f"       [WARNING] {key} has legacy BF16 lm_expert weights!")
        if len(cam_configs) > 1:
            raise ValueError(f"4 個模型的相機特徵不一致：{cam_configs}")
        if len(vlm_hashes) == 1:
            print("  -> All 4 checkpoints share 100% identical frozen VLM backbones.")
        else:
            print(f"  -> Note: {len(vlm_hashes)} distinct VLM backbones detected (supported automatically).")

    if args.benchmark or args.mock:
        print(f"\n=== 5-Stage Switching Benchmark (requested preload_mode={args.preload_mode}, mock={args.mock}) ===")
        t_init0 = time.perf_counter()
        pool = SmolVLAPolicyPool(
            checkpoint_map=ckpt_map,
            device=device,
            preload_mode=args.preload_mode,
            mock_models=args.mock,
        )
        pool.warmup()
        init_s = time.perf_counter() - t_init0
        print(f"Pool initialized (mode={pool.preload_mode}) + warmed up in {init_s:.2f}s")

        sequence = ["pick_cube", "put_cup", "pick_teabag", "put_cup", "shake_cup"]
        latencies: list[float] = []
        for idx, stage_key in enumerate(sequence, start=1):
            _, switch_ms = pool.activate(stage_key)
            latencies.append(switch_ms)
            print(f"  Stage {idx} -> {stage_key:12s}: {switch_ms:7.3f} ms")

        avg_ms = sum(latencies) / len(latencies)
        vram = pool.get_vram_mb()
        print(
            f"Average stage switch latency: {avg_ms:.3f} ms | "
            f"VRAM allocated={vram['allocated_mb']:.1f} MB | peak={vram['peak_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
