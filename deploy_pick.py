#!/usr/bin/env python
"""Deploy a fine-tuned SmolVLA policy on a real SO-101 follower arm.

Follows the MT05 inference pattern by loading ``SmolVLAPolicy`` together with
its saved ``preprocess`` and ``postprocess`` pipelines, reading dual OpenCV
cameras (``top`` and ``front``) and joint states, and sending predicted actions
to the SO-101 robot.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
from lerobot.utils.robot_utils import precise_sleep

HOMING_RAW_POS = 2048

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SmolVLA SO-101 deployment."""
    parser = argparse.ArgumentParser(description="Run SmolVLA inference on SO-101.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/smovla/checkpoints/latest",
        help="Path to the trained SmolVLA checkpoint directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the stylus from the stand",
        help="Natural language task prompt matching training.",
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="SO-101 serial port.")
    parser.add_argument("--robot-id", type=str, default="my_follower", help="SO-101 calibration id.")
    parser.add_argument("--top-cam", type=int, default=0, help="OpenCV camera index for 'top'.")
    parser.add_argument("--front-cam", type=int, default=2, help="OpenCV camera index for 'front'.")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width.")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency (Hz).")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="OpenCV FOURCC format (e.g. MJPG); pass empty string to auto-detect.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Rollout duration in seconds (<= 0 runs until Ctrl+C or 'q').",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Optional safety limit on per-step joint position delta (degrees).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=20,
        help="Steps executed per chunk before replanning (1..chunk_size; <=0 keeps checkpoint default).",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.4,
        help="EMA smoothing alpha in (0, 1] for joint commands (1.0 disables smoothing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live camera preview window (press 'q' to stop).",
    )
    parser.add_argument(
        "--home-duration",
        type=float,
        default=2.0,
        help="Seconds to smoothly return arm to its initial pose before disconnecting (default: 2.0).",
    )
    parser.add_argument(
        "--no-go-home",
        action="store_true",
        help="Skip returning to initial pose before disconnecting.",
    )
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="Keep motor torque enabled after disconnecting.",
    )
    return parser.parse_args()


def return_to_home(
    robot: SOFollower,
    target_pos: dict[str, int] | None = None,
    duration: float = 2.0,
    steps: int = 60,
) -> None:
    """Smoothly move the SO-101 arm back to target_pos (or raw 2048 if None)."""
    start_pos = robot.bus.sync_read("Present_Position", normalize=False)
    if target_pos is None:
        target_pos = {motor: HOMING_RAW_POS for motor in robot.bus.motors}
    dt = duration / steps if duration > 0 else 0.0
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = {
            motor: int(start_pos[motor] + alpha * (target_pos[motor] - start_pos[motor]))
            for motor in robot.bus.motors
        }
        robot.bus.sync_write("Goal_Position", interp, normalize=False)
        if dt > 0:
            time.sleep(dt)


def build_batch(obs: dict[str, object], task: str) -> dict[str, object]:
    """Convert an SO-101 observation dict into an MT05-style batched input dict."""
    state = np.array([obs[name] for name in JOINT_NAMES], dtype=np.float32)
    top_rgb = np.asarray(obs["top"])
    front_rgb = np.asarray(obs["front"])
    return {
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "observation.images.top": torch.from_numpy(top_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0)
        / 255.0,
        "observation.images.front": torch.from_numpy(front_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0)
        / 255.0,
        "task": [task],
    }


def main() -> None:
    """Load SmolVLA policy/processors and run the SO-101 closed-loop control loop."""
    args = parse_args()
    if not (0.0 < args.ema <= 1.0):
        raise ValueError(f"--ema must be in (0, 1], got {args.ema}")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    print(f"Loading SmolVLA checkpoint from {ckpt_path} onto {args.device}...")
    policy = SmolVLAPolicy.from_pretrained(ckpt_path).to(args.device).eval()
    if args.n_action_steps > 0:
        if args.n_action_steps > policy.config.chunk_size:
            raise ValueError(
                f"--n-action-steps ({args.n_action_steps}) cannot exceed chunk_size ({policy.config.chunk_size})."
            )
        policy.config.n_action_steps = args.n_action_steps
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.reset()

    fourcc = args.fourcc if args.fourcc else None
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=fourcc,
        ),
        "front": OpenCVCameraConfig(
            index_or_path=args.front_cam,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=fourcc,
        ),
    }
    robot_cfg = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=not args.keep_torque,
    )
    robot = SOFollower(robot_cfg)

    cv2 = None
    if args.display:
        import cv2 as _cv2

        cv2 = _cv2

    print("Connecting to SO-101 follower and cameras...")
    robot.connect()
    initial_pos = robot.bus.sync_read("Present_Position", normalize=False)
    print(
        f"Running policy for task: {args.task!r} at {args.fps} Hz "
        f"(n_action_steps={policy.config.n_action_steps}, ema={args.ema}). Press Ctrl+C to stop."
    )

    dt_target = 1.0 / args.fps
    start_t = time.perf_counter()
    step = 0
    smoothed_action: np.ndarray | None = None

    try:
        while True:
            if args.duration > 0 and (time.perf_counter() - start_t) >= args.duration:
                print(f"Reached duration limit ({args.duration:.1f}s). Stopping.")
                break

            loop_t0 = time.perf_counter()
            obs = robot.get_observation()
            batch = build_batch(obs, args.task)
            batch = preprocess(batch)

            with torch.inference_mode():
                action_tensor = postprocess(policy.select_action(batch))
            raw_action = action_tensor.squeeze(0).cpu().numpy()[: len(JOINT_NAMES)]
            if smoothed_action is None or args.ema >= 1.0:
                smoothed_action = raw_action.copy()
            else:
                smoothed_action = args.ema * raw_action + (1.0 - args.ema) * smoothed_action

            robot_action = {name: float(smoothed_action[i]) for i, name in enumerate(JOINT_NAMES)}
            robot.send_action(robot_action)

            if cv2 is not None:
                try:
                    preview = np.hstack([np.asarray(obs["top"]), np.asarray(obs["front"])])
                    cv2.imshow("SO-101 SmolVLA (top | front)", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped by user ('q').")
                        break
                except cv2.error as e:
                    print(f"Warning: disabling --display preview ({e})")
                    cv2 = None

            step += 1
            if step % args.fps == 0:
                elapsed = time.perf_counter() - start_t
                print(f"step {step:4d} | elapsed {elapsed:5.1f}s | action {np.round(smoothed_action, 2)}")

            precise_sleep(dt_target - (time.perf_counter() - loop_t0))
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        if robot.is_connected:
            if not args.no_go_home:
                try:
                    print("Returning SO-101 to initial pose...")
                    return_to_home(robot, target_pos=initial_pos, duration=max(0.0, args.home_duration))
                except Exception as e:
                    print(f"Warning: failed to return to initial pose before disconnect ({e})")
            robot.disconnect()
        if cv2 is not None:
            with contextlib.suppress(cv2.error):
                cv2.destroyAllWindows()
        print("Disconnected SO-101 cleanly.")


if __name__ == "__main__":
    main()
