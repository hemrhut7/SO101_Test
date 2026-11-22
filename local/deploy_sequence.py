"""Sequentially deploy multiple trained SO-101 checkpoints in one robot session.

Reuses PolicyRunner and robot helpers from deploy_local.py so cameras and the
SO-101 follower arm stay connected across model transitions, and smoothly returns
the arm to its initial pose before disconnecting.

Examples:
    # 1. Run two ACT models sequentially (e.g. so101_pick_cube -> so101_pick_teabag)
    uv run python -u local/deploy_sequence.py \
        --checkpoints \
            ./outputs/act/so101_pick_cube/checkpoints/latest \
            ./outputs/act/so101_pick_teabag/checkpoints/latest \
        --stage-steps 300 300 \
        --port /dev/ttyACM0 --robot-id my_follower \
        --cam-wrist 0 --cam-side-45 2 --display

    # 2. Return to initial pose and wait for Enter between independent tasks
    uv run python -u local/deploy_sequence.py \
        --checkpoints \
            ./outputs/act/so101_pick_cube/checkpoints/latest \
            ./outputs/act/so101_pick_teabag/checkpoints/latest \
        --stage-steps 300 300 --return-home-between --wait-enter-between \
        --port /dev/ttyACM0 --robot-id my_follower --display
"""

import argparse
import contextlib
import gc
import inspect
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deploy_local import (  # noqa: E402
    PolicyRunner,
    _parse_cam_source,
    detect_arch,
    return_to_home,
)
from local_utils import (  # noqa: E402
    DEPLOY_CUSTOM_REQUIRED_FILES,
    DEPLOY_LEROBOT_REQUIRED_FILES,
    verify_checkpoint_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequentially run multiple SO-101 checkpoints in one connected session."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Ordered list of checkpoint directories to run sequentially.",
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="Optional architectures per checkpoint (auto-detected if omitted).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task prompt per checkpoint (used when a stage model is SmolVLA).",
    )
    parser.add_argument(
        "--stage-steps",
        nargs="+",
        type=int,
        default=[300],
        help="Steps per checkpoint stage (1 value for all stages, or 1 per checkpoint; default: 300).",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Control loop frequency in Hz.")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Action chunk steps per forward pass (default: 35 SmolVLA/ACT, 15 FC/CNN).",
    )
    parser.add_argument(
        "--action-ema",
        "--ema",
        dest="action_ema",
        type=float,
        default=1.0,
        help="EMA smoothing factor in (0, 1] for consecutive actions.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cuda' or 'cpu').")
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 before Hub imports.")

    # Hardware & camera options
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for SO-101 follower.")
    parser.add_argument("--robot-id", default="my_follower", help="Robot ID for calibration file lookup.")
    parser.add_argument("--calibrate", action="store_true", help="Run/verify calibration on connect.")
    parser.add_argument("--no-use-degrees", action="store_true", help="Disable use_degrees=True.")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Safety clamp on per-step joint angle delta (default: 5.0).",
    )
    parser.add_argument("--home-duration", type=float, default=2.0, help="Smooth homing duration in seconds.")
    parser.add_argument("--no-go-home", action="store_true", help="Skip homing before disconnect.")
    parser.add_argument("--keep-torque", action="store_true", help="Keep motor torque after disconnect.")
    parser.add_argument(
        "--return-home-between",
        action="store_true",
        help="Smoothly return to initial pose between model stages.",
    )
    parser.add_argument(
        "--wait-enter-between",
        action="store_true",
        help="Pause and wait for Enter between model stages (useful for resetting objects).",
    )

    parser.add_argument("--cam-top", default="0", help="OpenCV index or path for top camera.")
    parser.add_argument("--cam-front", default="2", help="OpenCV index or path for front camera.")
    parser.add_argument("--cam-wrist", default="0", help="OpenCV index or path for wrist camera.")
    parser.add_argument("--cam-side-45", default="2", help="OpenCV index or path for side_45 camera.")
    parser.add_argument("--cam", nargs="*", default=[], metavar="NAME=SOURCE", help="Generic camera overrides.")
    parser.add_argument("--cam-width", type=int, default=640, help="Camera frame width.")
    parser.add_argument("--cam-height", type=int, default=480, help="Camera frame height.")
    parser.add_argument("--cam-warmup", type=int, default=3, help="Camera warmup timeout in seconds.")
    parser.add_argument("--fourcc", type=str, default="MJPG", help="OpenCV FOURCC format.")
    parser.add_argument("--no-cameras", action="store_true", help="Disable cameras (for state-only FC).")
    parser.add_argument("--display", action="store_true", help="Show live preview ('n'=next model, 'q'=quit).")
    parser.add_argument("--dry-run", action="store_true", help="Validate loading all models without hardware.")
    return parser.parse_args()


def _expand_per_stage(values, n_stages, flag_name, default_val=None):
    """Expand a 1-element or N-element CLI list to length n_stages."""
    if values is None:
        return [default_val] * n_stages
    if len(values) == 1:
        return [values[0]] * n_stages
    if len(values) == n_stages:
        return list(values)
    raise ValueError(f"{flag_name} 數量 ({len(values)}) 必須為 1 或與 --checkpoints ({n_stages}) 相同。")


def build_stage_specs(args):
    """Validate and resolve checkpoint paths, architectures, tasks, and steps for all stages."""
    n_stages = len(args.checkpoints)
    archs = _expand_per_stage(args.archs, n_stages, "--archs", default_val="auto")
    tasks = _expand_per_stage(args.tasks, n_stages, "--tasks", default_val="Pick up the stylus from the stand")
    steps_list = _expand_per_stage(args.stage_steps, n_stages, "--stage-steps")

    specs = []
    for idx, (raw_ckpt, raw_arch, task_text, stage_steps) in enumerate(
        zip(args.checkpoints, archs, tasks, steps_list)
    ):
        if stage_steps <= 0:
            raise ValueError(f"第 {idx + 1} 階段 --stage-steps 必須為正整數，目前為 {stage_steps}。")
        ckpt_dir = str(Path(raw_ckpt).expanduser().resolve())
        arch = detect_arch(ckpt_dir, raw_arch)
        is_lerobot = arch in ("smolvla", "act")
        req_files = DEPLOY_LEROBOT_REQUIRED_FILES if is_lerobot else DEPLOY_CUSTOM_REQUIRED_FILES
        exp_arch = "fc_mlp" if arch == "fc" else ("cnn" if arch == "cnn" else None)
        exp_type = arch if is_lerobot else None
        if not verify_checkpoint_dir(
            ckpt_dir,
            req_files,
            check_processor_states=is_lerobot,
            expected_arch=exp_arch,
            expected_type=exp_type,
        ):
            raise FileNotFoundError(f"Checkpoint 驗證失敗或不完整：{ckpt_dir}")
        specs.append(
            {
                "index": idx,
                "label": Path(raw_ckpt).parts[-3] if len(Path(raw_ckpt).parts) >= 3 else Path(ckpt_dir).name,
                "ckpt_dir": ckpt_dir,
                "arch": arch,
                "task": task_text,
                "steps": stage_steps,
            }
        )
    return specs


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps 必須為正數。")
    if not (0.0 < args.action_ema <= 1.0):
        raise ValueError("--action-ema 必須介於 (0.0, 1.0] 之間。")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import torch

    specs = build_stage_specs(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sequential deployment on {device} | total stages: {len(specs)}")
    for s in specs:
        print(f"  Stage {s['index'] + 1}: [{s['arch'].upper()}] {s['ckpt_dir']} | steps={s['steps']} | task={s['task']!r}")

    if args.dry_run:
        for s in specs:
            runner = PolicyRunner(s["arch"], s["ckpt_dir"], device, args.n_action_steps, action_ema=args.action_ema)
            print(f"[Dry-Run OK] Stage {s['index'] + 1} loaded | cameras={runner.camera_keys} | state_dim={runner.state_dim}")
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.robot_utils import precise_sleep

    # Inspect camera keys from the first model to configure SO-101 cameras once.
    first_runner = PolicyRunner(
        specs[0]["arch"], specs[0]["ckpt_dir"], device, args.n_action_steps, action_ema=args.action_ema
    )
    camera_keys = list(first_runner.camera_keys)
    if args.no_cameras and camera_keys:
        raise ValueError(f"模型需要相機輸入 {camera_keys}，不可搭配 --no-cameras。")

    cameras = {}
    if camera_keys:
        cam_sources = {
            "top": _parse_cam_source(args.cam_top),
            "front": _parse_cam_source(args.cam_front),
            "wrist": _parse_cam_source(args.cam_wrist),
            "side_45": _parse_cam_source(args.cam_side_45),
        }
        for item in args.cam:
            if "=" not in item:
                raise ValueError(f"--cam 格式應為 NAME=SOURCE，目前收到：{item}")
            name, src = item.split("=", 1)
            cam_sources[name.strip()] = _parse_cam_source(src.strip())
        fourcc = args.fourcc if args.fourcc else None
        for cam_key in camera_keys:
            cam_name = cam_key.split(".")[-1]
            if cam_name not in cam_sources:
                raise ValueError(f"模型需要相機 {cam_key}，可用 --cam {cam_name}=<編號或路徑> 指定。")
            cameras[cam_name] = OpenCVCameraConfig(
                index_or_path=cam_sources[cam_name],
                width=args.cam_width,
                height=args.cam_height,
                fps=int(args.fps),
                fourcc=fourcc,
                warmup_s=args.cam_warmup,
            )

    robot_kwargs = {"port": args.port, "id": args.robot_id, "cameras": cameras}
    if "use_degrees" in inspect.signature(SO101FollowerConfig).parameters:
        robot_kwargs["use_degrees"] = not args.no_use_degrees
    if "disable_torque_on_disconnect" in inspect.signature(SO101FollowerConfig).parameters:
        robot_kwargs["disable_torque_on_disconnect"] = not args.keep_torque
    if args.max_relative_target is not None:
        robot_kwargs["max_relative_target"] = args.max_relative_target

    robot = SO101Follower(SO101FollowerConfig(**robot_kwargs))
    cv2 = None
    if args.display and cameras:
        import cv2 as _cv2

        cv2 = _cv2

    period_s = 1.0 / max(args.fps, 1e-6)
    torch_device = torch.device(device)
    log_interval = max(1, int(args.fps))
    initial_pos = None

    try:
        print(f"Connecting SO-101 on {args.port} (id={args.robot_id}, cameras={list(cameras.keys())}) ...")
        robot.connect(calibrate=args.calibrate)
        try:
            initial_pos = robot.bus.sync_read("Present_Position", normalize=False)
        except Exception as e:
            print(f"Warning: failed to read initial pose ({e}); will fall back to raw 2048 on exit.")

        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        robot_type = getattr(robot, "robot_type", None)

        stop_requested = False
        for s in specs:
            if s["index"] == 0:
                runner = first_runner
                first_runner = None
            else:
                runner = PolicyRunner(
                    s["arch"], s["ckpt_dir"], device, args.n_action_steps, action_ema=args.action_ema
                )

            if list(runner.camera_keys) != camera_keys:
                raise ValueError(
                    f"第 {s['index'] + 1} 階段相機設定 {runner.camera_keys} 與第一階段 {camera_keys} 不一致。"
                )

            runner.reset()
            print(
                f"\n=== Stage {s['index'] + 1}/{len(specs)} | {s['label']} ({s['arch'].upper()}) "
                f"| steps={s['steps']} | task={s['task']!r} ==="
            )
            stage_start = time.perf_counter()
            for step in range(s["steps"]):
                loop_start = time.perf_counter()
                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=torch_device,
                    task=s["task"],
                    robot_type=robot_type,
                )
                action_tensor = runner.select_action(obs_frame)
                robot_action = make_robot_action(action_tensor, dataset_features)
                robot.send_action(robot_action)

                if cv2 is not None:
                    try:
                        frames = [np.asarray(obs[name]) for name in cameras if name in obs]
                        if frames:
                            bgr = cv2.cvtColor(np.hstack(frames), cv2.COLOR_RGB2BGR)
                            cv2.putText(
                                bgr,
                                f"Model {s['index'] + 1}/{len(specs)}: {s['label']}",
                                (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.imshow(f"SO-101 Sequence ({' | '.join(cameras.keys())})", bgr)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                print("Stopped by user ('q').")
                                stop_requested = True
                                break
                            if key == ord("n"):
                                print(f"Advancing to next model ('n' pressed at step {step}).")
                                break
                    except cv2.error as e:
                        print(f"Warning: disabling --display preview ({e})")
                        cv2 = None

                if step % log_interval == 0:
                    vals = [round(float(v), 2) for v in robot_action.values()]
                    elapsed = time.perf_counter() - stage_start
                    print(f"stage {s['index'] + 1} step {step:4d}/{s['steps']} | t={elapsed:5.1f}s | action: {vals}")

                precise_sleep(period_s - (time.perf_counter() - loop_start))

            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if stop_requested:
                break

            has_next = s["index"] + 1 < len(specs)
            if has_next and args.return_home_between:
                print("Returning to initial pose before next model...")
                return_to_home(robot, target_pos=initial_pos, duration=max(0.0, args.home_duration))
            if has_next and args.wait_enter_between:
                input("請重置桌面物件後按 Enter 繼續下一個模型...")
    except KeyboardInterrupt:
        print("\n[Stopped] Interrupted by user (Ctrl+C).")
    finally:
        if getattr(robot, "is_connected", True):
            if not args.no_go_home:
                try:
                    print("Returning SO-101 to initial pose...")
                    return_to_home(robot, target_pos=initial_pos, duration=max(0.0, args.home_duration))
                except Exception as home_err:
                    print(f"Warning: failed to return to initial pose ({home_err})")
            print("Disconnecting SO-101 robot ...")
            with contextlib.suppress(Exception):
                robot.disconnect()
        if cv2 is not None:
            with contextlib.suppress(cv2.error):
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
