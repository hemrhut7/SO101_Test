"""Closed-loop 5-stage continuous SO-101 deployment (YOLOE + SmolVLA).

Executes the full automated beverage preparation sequence:
    1. Pick up the teabag
    2. Put the teabag into the opaque cup
    3. Pick up the cube
    4. Put the cube into the opaque cup
    5. Shake the cup

Features:
- Preloads all 4 SmolVLA policies via ``SmolVLAPolicyPool`` (``gpu_full`` zero-copy
  on RTX 3060 12GB / AMD AI Max+ 395, or ``gpu_experts`` / ``cpu_pinned`` fallback).
- Runs in-process ``YOLOE`` visual-prompt monitoring on ``side_45`` (and ``wrist``)
  to verify grasps, opaque-cup placements, mid-flight drops, and timeout auto-resets.
- Supports ``--dry-run --mock-models`` for testing the complete FSM without hardware.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deploy_local import _parse_cam_source, return_to_home  # noqa: E402
from smolvla_pool import SmolVLAPolicyPool, VALID_PRELOAD_MODES  # noqa: E402
from yolo_monitor import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MockYOLOEPerceptionMonitor,
    StageVerificationStatus,
    YOLOEPerceptionMonitor,
)


@dataclass
class TaskStageSpec:
    index: int
    name: str
    policy_key: str
    task_prompt: str
    mode: str  # "pick" | "put" | "shake"
    target_object: str
    timeout_s: float
    min_steps: int
    return_home_after: bool
    rollback_stage_idx: int | None = None


def build_five_stage_pipeline(args: argparse.Namespace) -> list[TaskStageSpec]:
    """Construct the 5-stage SO-101 continuous drink preparation state machine."""
    return [
        TaskStageSpec(
            index=0,
            name="Stage 1: Pick Teabag",
            policy_key="pick_teabag",
            task_prompt=args.task_pick_teabag,
            mode="pick",
            target_object="tea bag",
            timeout_s=args.pick_timeout,
            min_steps=args.min_pick_steps,
            return_home_after=False,  # Keep holding teabag in mid-air for put_into_cup!
            rollback_stage_idx=None,
        ),
        TaskStageSpec(
            index=1,
            name="Stage 2: Put Teabag into Cup",
            policy_key="put_cup",
            task_prompt=args.task_put_cup,
            mode="put",
            target_object="tea bag",
            timeout_s=args.put_timeout,
            min_steps=args.min_put_steps,
            return_home_after=True,  # Return empty arm to Home before picking cube
            rollback_stage_idx=0,     # If teabag drops mid-flight, rollback to Stage 1
        ),
        TaskStageSpec(
            index=2,
            name="Stage 3: Pick Cube",
            policy_key="pick_cube",
            task_prompt=args.task_pick_cube,
            mode="pick",
            target_object="cube",
            timeout_s=args.pick_timeout,
            min_steps=args.min_pick_steps,
            return_home_after=False,  # Keep holding cube in mid-air for put_into_cup!
            rollback_stage_idx=None,
        ),
        TaskStageSpec(
            index=3,
            name="Stage 4: Put Cube into Cup",
            policy_key="put_cup",
            task_prompt=args.task_put_cup,
            mode="put",
            target_object="cube",
            timeout_s=args.put_timeout,
            min_steps=args.min_put_steps,
            return_home_after=True,  # Return empty arm to Home before shaking cup
            rollback_stage_idx=2,     # If cube drops mid-flight, rollback to Stage 3
        ),
        TaskStageSpec(
            index=4,
            name="Stage 5: Shake Cup",
            policy_key="shake_cup",
            task_prompt=args.task_shake_cup,
            mode="shake",
            target_object="cup",
            timeout_s=args.shake_duration,
            min_steps=int(args.shake_duration * args.fps),
            return_home_after=True,
            rollback_stage_idx=None,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SO-101 5-stage closed-loop deployment with SmolVLA + YOLOE."
    )
    # 4 SmolVLA Checkpoints & Task Prompts
    parser.add_argument(
        "--ckpt-pick-cube",
        default=str(PROJECT_ROOT / "outputs/smolvla/so101_pick_cube/checkpoints/latest"),
        help="Checkpoint path for 'pick up the cube'.",
    )
    parser.add_argument(
        "--ckpt-put-cup",
        default=str(PROJECT_ROOT / "outputs/smolvla/so101_put_into_cup/checkpoints/latest"),
        help="Checkpoint path for 'Pick the object into the cup'.",
    )
    parser.add_argument(
        "--ckpt-pick-teabag",
        default=str(PROJECT_ROOT / "outputs/smolvla/so101_pick_teabag/checkpoints/latest"),
        help="Checkpoint path for 'pick up the teabag'.",
    )
    parser.add_argument(
        "--ckpt-shake-cup",
        default=str(PROJECT_ROOT / "outputs/smolvla/so101_shake_cup/checkpoints/latest"),
        help="Checkpoint path for 'shake the cup'.",
    )
    parser.add_argument("--task-pick-cube", default="Pick up the cube")
    parser.add_argument("--task-put-cup", default="Pick the object into the cup")
    parser.add_argument("--task-pick-teabag", default="Pick up the teabag")
    parser.add_argument("--task-shake-cup", default="shake the cup")

    # SmolVLA Pool & Control Parameters
    parser.add_argument(
        "--preload-mode",
        default="auto",
        choices=list(VALID_PRELOAD_MODES),
        help="Model preloading strategy ('auto' chooses gpu_full on >=8GB VRAM like RTX 3060 / AI Max+ 395).",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Control loop frequency in Hz.")
    parser.add_argument("--n-action-steps", type=int, default=35, help="Action chunk horizon per forward pass.")
    parser.add_argument("--action-ema", "--ema", dest="action_ema", type=float, default=0.5, help="EMA smoothing factor.")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cuda' or 'cpu').")
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 before Hub imports.")

    # YOLOE Monitor & FSM Thresholds
    parser.add_argument("--yolo-weights", default="yoloe-11l-seg.pt", help="YOLOE weights path.")
    parser.add_argument("--yolo-config", default=str(DEFAULT_CONFIG_PATH), help="YOLOE visual prompt JSON config.")
    parser.add_argument("--yolo-mode", default="yoloe", choices=["yoloe", "standard"], help="YOLOE visual prompt or standard YOLO.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLOE confidence threshold.")
    parser.add_argument("--yolo-hz", type=float, default=10.0, help="Target YOLOE update rate in Hz (<= 0 falls back to --yolo-interval).")
    parser.add_argument("--yolo-interval", type=int, default=3, help="Fallback step interval when --yolo-hz <= 0.")
    parser.add_argument("--pick-timeout", type=float, default=40.0, help="Timeout (seconds) for pick stages before auto-reset.")
    parser.add_argument("--put-timeout", type=float, default=50.0, help="Timeout (seconds) for put stages before auto-reset.")
    parser.add_argument("--shake-duration", type=float, default=30.0, help="Duration (seconds) for shake the cup stage.")
    parser.add_argument("--min-pick-steps", type=int, default=300, help="Minimum steps before grasp verification triggers.")
    parser.add_argument("--min-put-steps", type=int, default=200, help="Minimum steps before cup release verification triggers.")
    parser.add_argument("--grasp-stable-checks", type=int, default=20, help="Consecutive YOLO checks to confirm grasp.")
    parser.add_argument("--put-stable-checks", type=int, default=20, help="Consecutive YOLO checks to confirm opaque-cup placement.")
    parser.add_argument("--lift-px", type=float, default=40.0, help="Upward pixel shift on side_45 to confirm lifted grasp.")
    parser.add_argument("--gripper-open-thresh", type=float, default=28.0, help="Gripper joint degree threshold indicating open jaw.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum auto-resets per stage before aborting.")

    # Hardware & Camera Options
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for SO-101 follower.")
    parser.add_argument("--robot-id", default="my_follower", help="Robot ID for calibration lookup.")
    parser.add_argument("--calibrate", action="store_true", help="Run/verify calibration on connect.")
    parser.add_argument("--no-use-degrees", action="store_true", help="Disable use_degrees=True.")
    parser.add_argument("--max-relative-target", type=float, default=5.0, help="Safety clamp on per-step joint angle delta.")
    parser.add_argument("--home-duration", type=float, default=1.8, help="Smooth homing duration in seconds.")
    parser.add_argument(
        "--put-start-pose",
        default="-8.8,-0.7,14.5,0.4,86.6",
        help="5-DOF arm pose (deg: pan,lift,elbow,wflex,wroll) to move to after 'pick' while keeping gripper closed; or 'home' / 'off'.",
    )
    parser.add_argument("--no-go-home", action="store_true", help="Skip homing before disconnect.")
    parser.add_argument("--keep-torque", action="store_true", help="Keep motor torque after disconnect.")
    parser.add_argument("--cam-wrist", default="0", help="OpenCV index/path for wrist camera.")
    parser.add_argument("--cam-side-45", default="2", help="OpenCV index/path for side_45 camera.")
    parser.add_argument("--cam", nargs="*", default=[], metavar="NAME=SOURCE", help="Generic camera overrides.")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-warmup", type=int, default=3)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--display", action="store_true", help="Show live HUD window ('n'=next, 'r'=reset stage, 'q'=quit).")

    # Offline / Development Simulation Options
    parser.add_argument("--dry-run", action="store_true", help="Run offline FSM simulation without connecting hardware.")
    parser.add_argument("--mock-models", action="store_true", help="Use mock SmolVLA + mock YOLOE (for dev laptop testing).")
    parser.add_argument(
        "--simulate-failures",
        action="store_true",
        help="With --dry-run --mock-models, inject Stage 1 timeout and Stage 2 mid-flight drop to verify recovery.",
    )
    return parser.parse_args()


def _release_gripper_if_closed(robot: Any, open_delta_raw: int = 250) -> None:
    """Slightly open the SO-101 gripper before homing during an error reset."""
    try:
        pos = robot.bus.sync_read("Present_Position", normalize=False)
        if "gripper" in pos:
            pos["gripper"] = min(4095, int(pos["gripper"]) + open_delta_raw)
            robot.bus.sync_write("Goal_Position", pos, normalize=False)
            time.sleep(0.25)
    except Exception:
        pass


def _parse_put_start_pose(raw: str) -> tuple[str, list[float] | None]:
    """Parse --put-start-pose into ('off'|'home'|'deg', optional 5-DOF degree list)."""
    text = str(raw).strip().lower()
    if text in ("off", "none", ""):
        return "off", None
    if text == "home":
        return "home", None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) != 5:
        raise ValueError(
            f"--put-start-pose 必須為 'off'、'home' 或 5 個逗號分隔的角度值 (pan,lift,elbow,wflex,wroll)，目前為 {raw!r}"
        )
    return "deg", [float(p) for p in parts]


def move_arm_preserve_gripper(
    robot: Any,
    mode: str,
    target_deg: list[float] | None = None,
    home_raw: dict[str, int] | None = None,
    duration: float = 1.5,
    steps: int = 45,
) -> None:
    """Smoothly move the 5 arm joints to a put-start pose while keeping the gripper closed."""
    if mode == "off":
        return
    dt = duration / steps if duration > 0 else 0.0
    if mode == "home":
        start_pos = robot.bus.sync_read("Present_Position", normalize=False)
        target = {
            m: (start_pos[m] if m == "gripper" else int((home_raw or {}).get(m, 2048)))
            for m in robot.bus.motors
        }
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = {
                m: int(start_pos[m] + alpha * (target[m] - start_pos[m]))
                for m in robot.bus.motors
            }
            robot.bus.sync_write("Goal_Position", interp, normalize=False)
            if dt > 0:
                time.sleep(dt)
        return

    if mode == "deg" and target_deg is not None:
        arm_motors = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        start_pos = robot.bus.sync_read("Present_Position", normalize=True)
        target_map = dict(zip(arm_motors, target_deg))
        gripper_hold = float(start_pos["gripper"]) if "gripper" in start_pos else None
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = {}
            for m in robot.bus.motors:
                if m in target_map and m in start_pos:
                    interp[m] = float(start_pos[m]) + alpha * (target_map[m] - float(start_pos[m]))
                elif m == "gripper" and gripper_hold is not None:
                    interp[m] = gripper_hold
            robot.bus.sync_write("Goal_Position", interp, normalize=True)
            if dt > 0:
                time.sleep(dt)


def run_dry_run_simulation(
    args: argparse.Namespace,
    pool: SmolVLAPolicyPool,
    stages: list[TaskStageSpec],
) -> None:
    """Verify complete 5-stage FSM, timeout auto-reset, and mid-flight drop rollback offline."""
    import numpy as np

    monitor = MockYOLOEPerceptionMonitor(
        grasp_stable_checks=args.grasp_stable_checks,
        put_stable_checks=args.put_stable_checks,
        simulate_failures=args.simulate_failures,
    )

    # Use fast step budget in dry-run simulation so tests complete in < 1 second
    sim_pick_min = 6
    sim_put_min = 5
    sim_shake_steps = 8
    sim_timeout_steps = 15

    print(
        f"\n[Dry-Run Simulation] preload_mode={pool.preload_mode} | "
        f"simulate_failures={args.simulate_failures} | stages={len(stages)}"
    )

    dummy_obs = {
        "observation.state": torch.zeros((1, pool.state_dim), dtype=torch.float32, device=pool.device),
        "wrist": np.zeros((args.cam_height, args.cam_width, 3), dtype=np.uint8),
        "side_45": np.zeros((args.cam_height, args.cam_width, 3), dtype=np.uint8),
    }
    monitor.capture_baseline(dummy_obs)

    retries = [0] * len(stages)
    stage_idx = 0
    total_transitions = 0

    while stage_idx < len(stages):
        spec = stages[stage_idx]
        runner, switch_ms = pool.activate(spec.policy_key)
        monitor.reset_stage_state()
        total_transitions += 1
        print(
            f"\n>>> Enter {spec.name} (attempt {retries[stage_idx] + 1}/{args.max_retries + 1}) "
            f"| switch={switch_ms:.3f} ms | task={spec.task_prompt!r}"
        )

        step = 0
        outcome = "timeout"
        while step < sim_timeout_steps:
            obs_frame = {
                "observation.state": dummy_obs["observation.state"],
                "task": spec.task_prompt,
            }
            for cam_key in pool.camera_keys:
                obs_frame[cam_key] = torch.zeros(
                    (1, 3, args.cam_height, args.cam_width), dtype=torch.float32, device=pool.device
                )
            _ = runner.select_action(obs_frame)

            if spec.mode == "pick":
                st = monitor.verify_grasp(spec.target_object, dummy_obs, step, min_pick_steps=sim_pick_min)
                if st.verified:
                    print(f"  [PASS] Grasp verified at step {step}: {st.reason}")
                    outcome = "done"
                    break
            elif spec.mode == "put":
                st = monitor.verify_put_in_opaque_cup(spec.target_object, dummy_obs, step, min_put_steps=sim_put_min)
                if st.dropped_on_table:
                    print(f"  [DROP] {st.reason}")
                    outcome = "dropped"
                    break
                if st.verified:
                    print(f"  [PASS] Opaque cup placement verified at step {step}: {st.reason}")
                    outcome = "done"
                    break
            elif spec.mode == "shake":
                if step + 1 >= sim_shake_steps:
                    print(f"  [PASS] Cup shaking completed ({sim_shake_steps} steps).")
                    outcome = "done"
                    break
            step += 1

        if outcome == "done":
            if spec.mode == "pick" and args.put_start_pose.strip().lower() not in ("off", "none", ""):
                print(f"  [Transition] Moving arm to put-start pose ({args.put_start_pose}) with gripper preserved.")
            if spec.return_home_after:
                print("  [Home] Smoothly returned to Home -> re-captured YOLOE baseline.")
                monitor.capture_baseline(dummy_obs)
            stage_idx += 1
        elif outcome == "dropped" and spec.rollback_stage_idx is not None:
            retries[spec.rollback_stage_idx] += 1
            if retries[spec.rollback_stage_idx] > args.max_retries:
                raise RuntimeError(f"Exceeded max retries during rollback to {stages[spec.rollback_stage_idx].name}.")
            print(
                f"  [Auto-Reset & Rollback] Returning Home -> updating baseline -> "
                f"rolling back to {stages[spec.rollback_stage_idx].name}"
            )
            monitor.capture_baseline(dummy_obs)
            stage_idx = spec.rollback_stage_idx
        else:
            retries[stage_idx] += 1
            if retries[stage_idx] > args.max_retries:
                raise RuntimeError(f"Stage {spec.name} failed after {args.max_retries} retries.")
            print(
                f"  [Timeout Auto-Reset] Returning Home -> flushing action queue -> "
                f"re-scanning baseline -> retrying {spec.name}"
            )
            monitor.capture_baseline(dummy_obs)

    print(f"\n[Dry-Run Success] Completed all 5 stages across {total_transitions} policy activations!")


def run_live_continuous_task(
    args: argparse.Namespace,
    pool: SmolVLAPolicyPool,
    stages: list[TaskStageSpec],
) -> None:
    """Connect SO-101 + dual cameras + YOLOE monitor and execute the 5-stage closed loop."""
    import numpy as np
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.robot_utils import precise_sleep

    monitor = YOLOEPerceptionMonitor(
        weights_path=args.yolo_weights,
        prompt_config_path=args.yolo_config,
        device=pool.device,
        conf=args.yolo_conf,
        yolo_interval=args.yolo_interval,
        yolo_hz=args.yolo_hz,
        grasp_stable_checks=args.grasp_stable_checks,
        put_stable_checks=args.put_stable_checks,
        lift_px=args.lift_px,
        gripper_open_thresh=args.gripper_open_thresh,
        yolo_mode=args.yolo_mode,
    )

    cam_sources = {
        "wrist": _parse_cam_source(args.cam_wrist),
        "side_45": _parse_cam_source(args.cam_side_45),
    }
    for item in args.cam:
        if "=" not in item:
            raise ValueError(f"--cam 格式應為 NAME=SOURCE，目前收到：{item}")
        name, src = item.split("=", 1)
        cam_sources[name.strip()] = _parse_cam_source(src.strip())

    fourcc = args.fourcc if args.fourcc else None
    cameras: dict[str, OpenCVCameraConfig] = {}
    for cam_key in pool.camera_keys:
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

    robot_kwargs: dict[str, Any] = {
        "port": args.port,
        "id": args.robot_id,
        "cameras": cameras,
    }
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
    torch_device = torch.device(pool.device)
    log_interval = max(1, int(args.fps))
    put_pose_mode, put_pose_deg = _parse_put_start_pose(args.put_start_pose)
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

        # Capture unoccluded tabletop baseline before Stage 1
        init_obs = robot.get_observation()
        baseline = monitor.capture_baseline(init_obs)
        print(f"[YOLOE Baseline] Detected objects: {list(baseline.keys())}")

        retries = [0] * len(stages)
        stage_idx = 0
        stop_requested = False

        while stage_idx < len(stages) and not stop_requested:
            spec = stages[stage_idx]
            runner, switch_ms = pool.activate(spec.policy_key)
            obs_now = robot.get_observation()
            joint_now = monitor._extract_joint_state(obs_now)
            monitor.reset_stage_state(joint_now)

            print(
                f"\n=== {spec.name} (Attempt {retries[stage_idx] + 1}/{args.max_retries + 1}) "
                f"| switch={switch_ms:.2f}ms | task={spec.task_prompt!r} ==="
            )

            stage_start = time.perf_counter()
            step = 0
            outcome = "timeout"

            while True:
                loop_start = time.perf_counter()
                elapsed = loop_start - stage_start
                if elapsed >= spec.timeout_s:
                    print(f"[Timeout] {spec.name} exceeded {spec.timeout_s:.1f}s.")
                    outcome = "timeout"
                    break

                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=torch_device,
                    task=spec.task_prompt,
                    robot_type=robot_type,
                )
                action_tensor = runner.select_action(obs_frame)
                robot_action = make_robot_action(action_tensor, dataset_features)
                robot.send_action(robot_action)

                status: StageVerificationStatus | None = None
                if spec.mode == "pick":
                    status = monitor.verify_grasp(spec.target_object, obs, step, min_pick_steps=spec.min_steps)
                    if status.verified:
                        print(f"[Verified] {spec.name} complete at step {step} ({status.reason}).")
                        outcome = "done"
                elif spec.mode == "put":
                    status = monitor.verify_put_in_opaque_cup(
                        spec.target_object, obs, step, min_put_steps=spec.min_steps
                    )
                    if status.dropped_on_table:
                        print(f"[Drop Detected] {spec.name}: {status.reason}")
                        outcome = "dropped"
                    elif status.verified:
                        print(f"[Verified] {spec.name} complete at step {step} ({status.reason}).")
                        outcome = "done"
                elif spec.mode == "shake":
                    monitor.update_detections(obs, step)
                    status = StageVerificationStatus(
                        verified=(step + 1 >= spec.min_steps),
                        stable_count=step + 1,
                        required_stable=spec.min_steps,
                        reason=f"shaking cup ({elapsed:.1f}/{spec.timeout_s:.1f}s)",
                        detections=dict(monitor.latest_detections),
                    )
                    if status.verified:
                        print(f"[Verified] {spec.name} complete ({elapsed:.1f}s).")
                        outcome = "done"

                if cv2 is not None:
                    try:
                        hud = monitor.draw_hud(
                            obs=obs,
                            stage_idx=stage_idx,
                            total_stages=len(stages),
                            stage_label=spec.name,
                            switch_ms=switch_ms,
                            elapsed_s=elapsed,
                            timeout_s=spec.timeout_s,
                            retries=retries[stage_idx],
                            status=status,
                        )
                        if hud is not None:
                            cv2.imshow("SO-101 Closed-Loop (YOLOE + SmolVLA)", hud)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                print("Stopped by user ('q').")
                                stop_requested = True
                                outcome = "abort"
                                break
                            if key == ord("n"):
                                print(f"Force advancing to next stage ('n' pressed at step {step}).")
                                outcome = "done"
                                break
                            if key == ord("r"):
                                print(f"Manual reset requested ('r' pressed at step {step}).")
                                outcome = "timeout"
                                break
                    except cv2.error as e:
                        print(f"Warning: disabling --display preview ({e})")
                        cv2 = None

                if outcome in ("done", "dropped"):
                    break

                if step % log_interval == 0:
                    msg_status = status.reason if status is not None else "running"
                    print(f"  step {step:4d} | t={elapsed:5.1f}/{spec.timeout_s:.0f}s | {msg_status}")

                step += 1
                precise_sleep(period_s - (time.perf_counter() - loop_start))

            if stop_requested or outcome == "abort":
                break

            if outcome == "done":
                if spec.mode == "pick" and put_pose_mode != "off":
                    print(f"Moving arm smoothly to put-start pose ({args.put_start_pose}) while keeping gripper closed...")
                    move_arm_preserve_gripper(
                        robot,
                        mode=put_pose_mode,
                        target_deg=put_pose_deg,
                        home_raw=initial_pos,
                        duration=max(0.5, args.home_duration * 0.8),
                    )
                if spec.return_home_after:
                    print("Returning arm smoothly to Home pose before next stage...")
                    return_to_home(robot, target_pos=initial_pos, duration=max(0.0, args.home_duration))
                    fresh_obs = robot.get_observation()
                    monitor.capture_baseline(fresh_obs)
                stage_idx += 1
            elif outcome == "dropped" and spec.rollback_stage_idx is not None:
                target_idx = spec.rollback_stage_idx
                retries[target_idx] += 1
                if retries[target_idx] > args.max_retries:
                    print(f"[Abort] Exceeded --max-retries ({args.max_retries}) on {stages[target_idx].name}.")
                    break
                print(
                    f"[Auto-Reset & Rollback] Homing arm and rolling back to {stages[target_idx].name} "
                    f"(retry {retries[target_idx]}/{args.max_retries})..."
                )
                _release_gripper_if_closed(robot)
                return_to_home(robot, target_pos=initial_pos, duration=max(0.0, args.home_duration))
                fresh_obs = robot.get_observation()
                monitor.capture_baseline(fresh_obs)
                stage_idx = target_idx
            else:
                retries[stage_idx] += 1
                if retries[stage_idx] > args.max_retries:
                    print(f"[Abort] {spec.name} failed after {args.max_retries} retries.")
                    break
                print(
                    f"[Timeout Auto-Reset] Homing arm and re-scanning table for {spec.name} "
                    f"(retry {retries[stage_idx]}/{args.max_retries})..."
                )
                _release_gripper_if_closed(robot)
                return_to_home(robot, target_pos=initial_pos, duration=max(0.0, args.home_duration))
                fresh_obs = robot.get_observation()
                monitor.capture_baseline(fresh_obs)

        if stage_idx >= len(stages) and not stop_requested:
            print("\n=== ALL 5 STAGES COMPLETED SUCCESSFULLY ===")

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


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps 必須為正數。")
    if not (0.0 < args.action_ema <= 1.0):
        raise ValueError("--action-ema 必須介於 (0.0, 1.0] 之間。")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_map = {
        "pick_cube": args.ckpt_pick_cube,
        "put_cup": args.ckpt_put_cup,
        "pick_teabag": args.ckpt_pick_teabag,
        "shake_cup": args.ckpt_shake_cup,
    }
    stages = build_five_stage_pipeline(args)

    print(f"Initializing SmolVLAPolicyPool on {device} (requested preload_mode={args.preload_mode}) ...")
    pool = SmolVLAPolicyPool(
        checkpoint_map=checkpoint_map,
        device=device,
        preload_mode=args.preload_mode,
        n_action_steps=args.n_action_steps,
        action_ema=args.action_ema,
        mock_models=args.mock_models,
    )
    print(f"Pool ready | active preload_mode={pool.preload_mode} | unique models={len(pool.unique_paths)}")

    if not args.mock_models:
        print("Warming up SmolVLA policies on target device...")
        pool.warmup(task_text=args.task_pick_teabag)
        vram = pool.get_vram_mb()
        if vram["allocated_mb"] > 0:
            print(f"GPU VRAM after preload+warmup: allocated={vram['allocated_mb']:.1f} MB, peak={vram['peak_mb']:.1f} MB")

    if args.dry_run:
        run_dry_run_simulation(args, pool, stages)
    else:
        run_live_continuous_task(args, pool, stages)


if __name__ == "__main__":
    main()
