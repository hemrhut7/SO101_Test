"""Local SO-101 deployment and inference script for SmolVLA, ACT, FC (MLP), and CNN.

Examples:
    # 1. Offline dry-run (no robot hardware required)
    python local/deploy_local.py --checkpoint ./outputs/act/checkpoints/latest --dry-run

    # 2. Live SO-101 robot deployment (Windows COM port)
    python local/deploy_local.py --checkpoint ./outputs/smolvla/checkpoints/latest ^
        --port COM5 --robot-id my_so101 --cam-top 0 --cam-front 1 ^
        --task "Pick up the stylus" --offline

    # 3. Live SO-101 robot deployment with ACT, FC, or CNN
    python local/deploy_local.py --checkpoint ./outputs/act/checkpoints/latest ^
        --port COM5 --robot-id my_so101 --cam-top 0 --cam-front 1 --n-action-steps 15
"""

import argparse
from collections import deque
import contextlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy trained SmolVLA / ACT / FC / CNN policy on SO-101.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (e.g., ./outputs/cnn/checkpoints/latest).",
    )
    parser.add_argument(
        "--arch",
        default="auto",
        choices=["auto", "smolvla", "act", "fc", "cnn"],
        help="Model architecture (default: auto-detect from config.json).",
    )
    parser.add_argument(
        "--task",
        default="Pick up the stylus from the stand",
        help="Language task instruction (for SmolVLA).",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Control loop frequency in Hz.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of deployment episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode (<= 0 allowed when --duration > 0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional total rollout duration limit in seconds (<= 0 disables time limit).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Action chunk steps per forward pass (default: 35 for SmolVLA/ACT, 15 for FC/CNN; <= 0 keeps checkpoint default).",
    )
    parser.add_argument(
        "--action-ema",
        "--ema",
        dest="action_ema",
        type=float,
        default=1.0,
        help="EMA smoothing factor in (0, 1] for consecutive actions (1.0 = no smoothing; e.g. 0.4 or 0.5 smooths jitter).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda' or 'cpu'); defaults to auto-detect.",
    )
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 before Hub imports.")

    # Robot hardware options
    parser.add_argument("--port", default=None, help="Serial port for SO-101 follower (e.g., COM5 or /dev/ttyACM0).")
    parser.add_argument("--robot-id", default="so101_follower", help="Robot ID for calibration file lookup.")
    parser.add_argument("--calibrate", action="store_true", help="Run/verify calibration when connecting robot.")
    parser.add_argument(
        "--no-use-degrees",
        action="store_true",
        help="Disable use_degrees=True in SO101FollowerConfig (not recommended for datasets recorded in degrees).",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Optional safety clamp on per-step joint angle delta.",
    )
    parser.add_argument("--cam-top", default="0", help="OpenCV index or device path for top camera.")
    parser.add_argument("--cam-front", default="1", help="OpenCV index or device path for front camera.")
    parser.add_argument("--cam-width", type=int, default=640, help="Camera frame width.")
    parser.add_argument("--cam-height", type=int, default=480, help="Camera frame height.")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="OpenCV FOURCC format (e.g. MJPG); pass empty string to auto-detect.",
    )
    parser.add_argument("--no-cameras", action="store_true", help="Disable cameras (for state-only FC policy).")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live camera preview window during live deployment (press 'q' to stop).",
    )

    # Dry-run options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference without connecting physical SO-101 hardware.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(PROJECT_ROOT / "dataset"),
        help="Optional dataset directory for --dry-run replay (falls back to synthetic tensors if missing).",
    )
    parser.add_argument("--repo-id", default="local/so101_pick_stylus", help="LeRobot dataset repo_id for --dry-run.")
    parser.add_argument("--video-backend", default="pyav", help="Video backend when --dry-run reads a dataset.")
    return parser.parse_args()


def detect_arch(ckpt_dir, requested_arch):
    if requested_arch != "auto":
        return requested_arch
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Checkpoint 目錄缺少 config.json：{ckpt_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    arch = cfg.get("arch")
    if arch == "fc_mlp":
        return "fc"
    if arch == "cnn":
        return "cnn"
    cfg_type = cfg.get("type")
    if cfg_type == "act":
        return "act"
    if cfg_type == "smolvla":
        return "smolvla"
    if os.path.isfile(os.path.join(ckpt_dir, "model.safetensors")):
        return "smolvla"
    raise ValueError(f"無法從 {ckpt_dir} 自動辨識模型架構，請手動指定 --arch。")


def _parse_cam_source(val):
    val_str = str(val).strip()
    return int(val_str) if val_str.isdigit() else val_str


class PolicyRunner:
    """Unified runtime wrapper for SmolVLA, ACT, FC (MLP), and CNN action chunk inference."""

    def __init__(self, arch, ckpt_dir, device, n_action_steps, action_ema=1.0):
        self.arch = arch
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.n_action_steps = int(n_action_steps) if n_action_steps is not None else (35 if arch in ("smolvla", "act") else 15)
        self.action_ema = float(action_ema)
        self._prev_action = None
        self.action_queue = deque()
        self.preprocess = None
        self.postprocess = None

        if arch in ("smolvla", "act"):
            from lerobot.policies.factory import make_pre_post_processors

            if arch == "smolvla":
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                from local_utils import is_legacy_bf16_smolvla_checkpoint

                if is_legacy_bf16_smolvla_checkpoint(ckpt_dir):
                    print(
                        "[WARNING] 此 SmolVLA checkpoint 的 lm_expert 權重為舊版 BF16"
                        "（受梯度捨入消失影響，33 個 LayerNorm 未更新），實機部署可能出現上下抽搐！"
                        "建議使用修正後的腳本重新訓練。"
                    )
                self.policy = SmolVLAPolicy.from_pretrained(ckpt_dir).to(device).eval()
            else:
                from lerobot.policies.act.modeling_act import ACTPolicy

                self.policy = ACTPolicy.from_pretrained(ckpt_dir).to(device).eval()

            if self.n_action_steps > 0 and hasattr(self.policy.config, "n_action_steps"):
                if getattr(self.policy.config, "temporal_ensemble_coeff", None) is not None:
                    print("Note: checkpoint uses temporal_ensemble_coeff; --n-action-steps queue override does not apply.")
                max_chunk = getattr(self.policy.config, "chunk_size", self.n_action_steps)
                self.policy.config.n_action_steps = min(self.n_action_steps, max_chunk)
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config,
                pretrained_path=ckpt_dir,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            self.camera_keys = [
                k for k in self.policy.config.input_features if k.startswith("observation.images.")
            ]
            self.state_dim = self.policy.config.input_features["observation.state"].shape[0]
        else:
            from local_policies import load_custom_policy_from_checkpoint

            self.policy = load_custom_policy_from_checkpoint(ckpt_dir, device)
            self.policy.eval()
            effective_steps = self.n_action_steps if self.n_action_steps > 0 else self.policy.chunk_size
            self.n_action_steps = max(1, min(effective_steps, self.policy.chunk_size))
            self.camera_keys = list(self.policy.camera_keys) if getattr(self.policy, "use_images", True) else []
            self.state_dim = self.policy.state_dim

        self.reset()

    def reset(self):
        self.action_queue.clear()
        self._prev_action = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def select_action(self, obs_frame):
        import torch

        if self.arch in ("smolvla", "act"):
            processed = self.preprocess(obs_frame)
            with torch.inference_mode():
                action = self.policy.select_action(processed)
                action = self.postprocess(action)
        else:
            if not self.action_queue:
                chunk = self.policy.predict_action(obs_frame)  # (1, CHUNK, action_dim)
                for i in range(self.n_action_steps):
                    self.action_queue.append(chunk[:, i])
            action = self.action_queue.popleft()

        if 0.0 < self.action_ema < 1.0:
            if self._prev_action is not None:
                action = self.action_ema * action + (1.0 - self.action_ema) * self._prev_action
            self._prev_action = action.detach().clone()
        return action


def run_dry_run(args, runner, device):
    import numpy as np
    import torch

    dataset_samples = None
    ds = None
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from local_utils import resolve_dataset_root

        ds_root = resolve_dataset_root(args.dataset_dir)
        ds = LeRobotDataset(args.repo_id, root=ds_root, video_backend=args.video_backend)
        dataset_samples = min(args.max_steps, len(ds))
        print(f"[Dry-Run] Using dataset from {ds_root} ({dataset_samples} steps)")
    except Exception as e:
        print(f"[Dry-Run] Dataset unavailable ({e}); using synthetic tensors instead.")

    runner.reset()
    latencies_ms = []
    num_steps = dataset_samples if dataset_samples is not None else min(args.max_steps, 10)

    for step in range(num_steps):
        if ds is not None:
            raw = ds[step]
            obs_frame = {}
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    obs_frame[k] = v.unsqueeze(0).to(device)
                else:
                    obs_frame[k] = v
            if runner.arch == "smolvla" and "task" not in obs_frame:
                obs_frame["task"] = args.task
            gt_action = raw.get("action")
        else:
            obs_frame = {
                "observation.state": torch.zeros((1, runner.state_dim), dtype=torch.float32, device=device),
                "task": args.task,
            }
            for cam in runner.camera_keys:
                obs_frame[cam] = torch.zeros(
                    (1, 3, args.cam_height, args.cam_width), dtype=torch.float32, device=device
                )
            gt_action = None

        t0 = time.perf_counter()
        pred = runner.select_action(obs_frame)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        pred_np = pred.squeeze(0).detach().cpu().numpy()
        if step < 5 or step == num_steps - 1:
            msg = f"step {step:3d} | latency {dt_ms:6.2f} ms | pred: {np.round(pred_np, 3)}"
            if gt_action is not None:
                gt_np = gt_action.detach().cpu().numpy()
                mae = float(np.mean(np.abs(pred_np - gt_np)))
                msg += f" | gt_mae: {mae:.3f}"
            print(msg)

    avg_ms = sum(latencies_ms) / max(1, len(latencies_ms))
    print(f"[Dry-Run Done] avg latency: {avg_ms:.2f} ms ({1000.0 / max(avg_ms, 1e-6):.1f} Hz)")


def run_live_robot(args, runner, device):
    import numpy as np
    import torch
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.robot_utils import precise_sleep

    if not args.port:
        raise ValueError("實機部署必須提供 --port（例如 Windows: COM5，Linux: /dev/ttyACM0），或加上 --dry-run 進行離線測試。")
    if args.no_cameras and runner.camera_keys:
        raise ValueError(f"模型需要相機輸入 {runner.camera_keys}，不可搭配 --no-cameras。")
    if args.fps < 1.0:
        raise ValueError(f"實機部署搭配 OpenCV 相機時 --fps 須 >= 1.0，目前為 {args.fps}。")

    fourcc = args.fourcc if args.fourcc else None
    cameras = {}
    if runner.camera_keys:
        cam_sources = {
            "top": _parse_cam_source(args.cam_top),
            "front": _parse_cam_source(args.cam_front),
        }
        for cam_key in runner.camera_keys:
            cam_name = cam_key.split(".")[-1]
            if cam_name not in cam_sources:
                raise ValueError(f"模型需要相機 {cam_key}，但未設定對應來源。")
            cameras[cam_name] = OpenCVCameraConfig(
                index_or_path=cam_sources[cam_name],
                width=args.cam_width,
                height=args.cam_height,
                fps=int(args.fps),
                fourcc=fourcc,
            )

    robot_kwargs = {
        "port": args.port,
        "id": args.robot_id,
        "cameras": cameras,
    }
    if "use_degrees" in inspect.signature(SO101FollowerConfig).parameters:
        robot_kwargs["use_degrees"] = not args.no_use_degrees
    if args.max_relative_target is not None:
        robot_kwargs["max_relative_target"] = args.max_relative_target

    robot_cfg = SO101FollowerConfig(**robot_kwargs)
    robot = SO101Follower(robot_cfg)

    cv2 = None
    if args.display and cameras:
        import cv2 as _cv2

        cv2 = _cv2
    elif args.display and not cameras:
        print("Note: --display ignored because no cameras are active.")

    period_s = 1.0 / max(args.fps, 1e-6)
    torch_device = torch.device(device)
    log_interval = max(1, int(args.fps))

    try:
        print(f"Connecting SO-101 on {args.port} (id={args.robot_id}, cameras={list(cameras.keys())}) ...")
        robot.connect(calibrate=args.calibrate)

        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        robot_type = getattr(robot, "robot_type", None)

        rollout_start = time.perf_counter()
        stop_requested = False

        for ep in range(args.episodes):
            runner.reset()
            print(
                f"\n=== Episode {ep + 1}/{args.episodes} | task: '{args.task}' "
                f"| fps={args.fps} | action_ema={args.action_ema} ==="
            )
            step = 0
            while args.max_steps <= 0 or step < args.max_steps:
                if args.duration > 0 and (time.perf_counter() - rollout_start) >= args.duration:
                    print(f"Reached duration limit ({args.duration:.1f}s). Stopping.")
                    stop_requested = True
                    break

                loop_start = time.perf_counter()

                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=torch_device,
                    task=args.task,
                    robot_type=robot_type,
                )
                action_tensor = runner.select_action(obs_frame)
                robot_action = make_robot_action(action_tensor, dataset_features)
                robot.send_action(robot_action)

                if cv2 is not None:
                    try:
                        frames = [np.asarray(obs[name]) for name in cameras if name in obs]
                        if frames:
                            preview = np.hstack(frames)
                            title = f"SO-101 {runner.arch.upper()} ({' | '.join(cameras.keys())})"
                            cv2.imshow(title, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                print("Stopped by user ('q').")
                                stop_requested = True
                                break
                    except cv2.error as e:
                        print(f"Warning: disabling --display preview ({e})")
                        cv2 = None

                if step % log_interval == 0:
                    vals = [round(float(v), 2) for v in robot_action.values()]
                    step_label = f"{step:4d}/{args.max_steps}" if args.max_steps > 0 else f"{step:4d}"
                    elapsed_total = time.perf_counter() - rollout_start
                    print(f"ep {ep + 1} step {step_label} | t={elapsed_total:5.1f}s | action: {vals}")

                step += 1
                precise_sleep(period_s - (time.perf_counter() - loop_start))

            if stop_requested:
                break
    except KeyboardInterrupt:
        print("\n[Stopped] Deployment interrupted by user (Ctrl+C).")
    finally:
        if getattr(robot, "is_connected", True):
            print("Disconnecting SO-101 robot ...")
            try:
                robot.disconnect()
                print("Robot disconnected safely.")
            except Exception as disc_err:
                print(f"Warning: disconnect raised an error: {disc_err}")
        if cv2 is not None:
            with contextlib.suppress(cv2.error):
                cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.fps <= 0 or args.episodes <= 0:
        raise ValueError("--fps 與 --episodes 必須皆為正數。")
    if args.max_steps <= 0 and args.duration <= 0:
        raise ValueError("--max-steps 必須為正數（除非已設定 --duration > 0）。")
    if not (0.0 < args.action_ema <= 1.0):
        raise ValueError("--action-ema / --ema 必須介於 (0.0, 1.0] 之間。")

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    ckpt_dir = str(Path(args.checkpoint).expanduser().resolve())
    arch = detect_arch(ckpt_dir, args.arch)

    from local_utils import DEPLOY_CUSTOM_REQUIRED_FILES, DEPLOY_LEROBOT_REQUIRED_FILES, verify_checkpoint_dir

    is_lerobot_policy = arch in ("smolvla", "act")
    required_files = DEPLOY_LEROBOT_REQUIRED_FILES if is_lerobot_policy else DEPLOY_CUSTOM_REQUIRED_FILES
    expected_arch = "fc_mlp" if arch == "fc" else ("cnn" if arch == "cnn" else None)
    expected_type = arch if is_lerobot_policy else None

    if not verify_checkpoint_dir(
        ckpt_dir,
        required_files,
        check_processor_states=is_lerobot_policy,
        expected_arch=expected_arch,
        expected_type=expected_type,
    ):
        raise FileNotFoundError(f"Checkpoint 驗證失敗或不完整：{ckpt_dir}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {arch.upper()} checkpoint from {ckpt_dir} on {device} ...")
    runner = PolicyRunner(arch, ckpt_dir, device, args.n_action_steps, action_ema=args.action_ema)
    print(f"Model ready | cameras={runner.camera_keys} | state_dim={runner.state_dim}")

    if args.dry_run:
        run_dry_run(args, runner, device)
    else:
        run_live_robot(args, runner, device)


if __name__ == "__main__":
    main()
