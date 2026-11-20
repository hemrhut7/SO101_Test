#!/usr/bin/env python
"""Smoothly move an SO-101 follower arm back to its initial calibration pose (raw 2048)."""

from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

HOMING_RAW_POS = 2048


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for returning SO-101 to its homing position."""
    parser = argparse.ArgumentParser(description="Move SO-101 back to initial calibration position.")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="SO-101 serial port.")
    parser.add_argument("--robot-id", type=str, default="my_follower", help="SO-101 calibration id.")
    parser.add_argument("--duration", type=float, default=2.0, help="Motion duration in seconds.")
    parser.add_argument("--steps", type=int, default=60, help="Number of interpolation steps.")
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="Keep motor torque enabled after disconnecting.",
    )
    return parser.parse_args()


def main() -> None:
    """Connect to SO-101 and interpolate all joints to raw encoder position 2048."""
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.duration < 0:
        raise ValueError("--duration must be non-negative.")

    config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        disable_torque_on_disconnect=not args.keep_torque,
    )
    robot = SOFollower(config)
    robot.connect(calibrate=False)

    try:
        start_pos = robot.bus.sync_read("Present_Position", normalize=False)
        target_pos = {motor: HOMING_RAW_POS for motor in robot.bus.motors}

        dt = args.duration / args.steps if args.duration > 0 else 0.0
        for i in range(1, args.steps + 1):
            alpha = i / args.steps
            interp = {
                motor: int(start_pos[motor] + alpha * (target_pos[motor] - start_pos[motor]))
                for motor in robot.bus.motors
            }
            robot.bus.sync_write("Goal_Position", interp, normalize=False)
            if dt > 0:
                time.sleep(dt)

        print(f"SO-101 ({args.robot_id}) returned to initial calibration position ({HOMING_RAW_POS}).")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
