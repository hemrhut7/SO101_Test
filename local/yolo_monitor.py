"""In-process YOLOE visual-prompt monitor and opaque-cup state verification for SO-101.

Built directly around YOLOE visual prompting (reference image + 3 reference bounding
boxes for ``tea bag``, ``cube``, and ``cup``) without requiring YOLO fine-tuning.
Shares frames directly from ``robot.get_observation()`` to avoid V4L2 camera lock
conflicts and runs decimated inference (default 10 Hz) so the 30 Hz control loop
never drops frames.

Standalone reference box calibration:
    uv run python -u local/yolo_monitor.py --calibrate-ref --cam-side-45 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "yoloe_prompt_config.json"
DEFAULT_REF_IMAGE_PATH = SCRIPT_DIR / "yoloe_ref.jpg"

DEFAULT_NAMES = ["tea bag", "cube", "cup"]
DEFAULT_COLORS = {
    "tea bag": (0, 200, 0),
    "cube": (255, 120, 0),
    "cup": (0, 140, 255),
}


def resolve_yolo_device(requested_device: str | None = None) -> str:
    """Select CUDA/ROCm when available, else MPS, else CPU."""
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _point_in_roi(pt: tuple[float, float], roi: list[float]) -> bool:
    return roi[0] <= pt[0] <= roi[2] and roi[1] <= pt[1] <= roi[3]


def _boxes_overlap(a: list[float], b: list[float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


@dataclass
class DetectionBox:
    cls_name: str
    bbox: list[float]  # [x1, y1, x2, y2] in pixels
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return _bbox_center(self.bbox)


@dataclass
class StageVerificationStatus:
    verified: bool = False
    dropped_on_table: bool = False
    stable_count: int = 0
    required_stable: int = 1
    entered_cup_zone: bool = False
    gripper_released: bool = False
    reason: str = "initializing"
    detections: dict[str, DetectionBox] = field(default_factory=dict)


def load_yoloe_prompt_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load saved YOLOE reference image and bounding boxes, or fall back to webcam.py defaults."""
    cfg_path = Path(config_path)
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ref_path = Path(data["ref_image"])
        if not ref_path.is_absolute():
            ref_path = (cfg_path.parent / ref_path).resolve()
        data["ref_image"] = str(ref_path)
        return data

    # Fallback to root webcam.py default image if present
    fallback_ref = PROJECT_ROOT / "frame_093640.jpg"
    return {
        "ref_image": str(fallback_ref),
        "names": list(DEFAULT_NAMES),
        "boxes": [
            [790, 618, 1000, 738],
            [442, 501, 607, 673],
            [688, 305, 894, 617],
        ],
    }


def calibrate_yoloe_reference(
    cam_source: int | str = 2,
    image_path: str | None = None,
    out_config: str | Path = DEFAULT_CONFIG_PATH,
    out_ref_image: str | Path = DEFAULT_REF_IMAGE_PATH,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    """Capture a reference frame from side_45 and interactively select ROI boxes for YOLOE."""
    frame_bgr: np.ndarray | None = None
    if image_path:
        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise FileNotFoundError(f"無法讀取影像檔：{image_path}")
    else:
        src = int(cam_source) if str(cam_source).isdigit() else str(cam_source)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟相機 {cam_source!r}，請確認相機編號或是否被其他程式佔用。")
        print("相機預覽中：請將桌面物件（tea bag / cube / cup）擺好並移開手臂，按 's' 擷取參考畫面，按 'q' 取消。")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                preview = frame.copy()
                cv2.putText(
                    preview,
                    "Press 's' to snapshot reference frame | 'q' to quit",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("YOLOE Reference Snapshot", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    frame_bgr = frame.copy()
                    break
                if key == ord("q"):
                    raise KeyboardInterrupt("User cancelled calibration.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    assert frame_bgr is not None
    out_img_path = Path(out_ref_image).resolve()
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), frame_bgr)
    print(f"已儲存參考畫面：{out_img_path}")

    boxes: list[list[int]] = []
    for name in DEFAULT_NAMES:
        win_title = f"Select '{name}' ROI then press ENTER/SPACE (ESC to skip)"
        x, y, w, h = cv2.selectROI(win_title, frame_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win_title)
        if w <= 0 or h <= 0:
            raise ValueError(f"未選擇 '{name}' 的有效區域，請重新執行校準。")
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
        print(f"  {name}: {boxes[-1]}")

    out_cfg_path = Path(out_config).resolve()
    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel_ref = str(out_img_path.relative_to(out_cfg_path.parent))
    except ValueError:
        rel_ref = str(out_img_path)

    payload = {
        "ref_image": rel_ref,
        "names": list(DEFAULT_NAMES),
        "boxes": boxes,
    }
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"已儲存 YOLOE 視覺提示設定：{out_cfg_path}")
    return payload


class YOLOEPerceptionMonitor:
    """In-process YOLOE monitor providing grasp, opaque-cup placement, and HUD tracking."""

    def __init__(
        self,
        weights_path: str = "yoloe-11l-seg.pt",
        prompt_config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str | None = None,
        conf: float = 0.25,
        yolo_interval: int = 3,
        side_cam_key: str = "side_45",
        wrist_cam_key: str = "wrist",
        grasp_stable_checks: int = 5,
        put_stable_checks: int = 8,
        drop_confirm_checks: int = 5,
        lift_px: float = 40.0,
        cup_rim_pad_x: float = 28.0,
        cup_rim_height: float = 85.0,
        gripper_open_thresh: float = 28.0,
        gripper_open_delta: float = 10.0,
        yolo_mode: str = "yoloe",
    ) -> None:
        try:
            from ultralytics import YOLO, YOLOE
        except ImportError as exc:
            raise ImportError(
                "尚未安裝 'ultralytics'。請先執行 `uv add ultralytics` 或 "
                "`uv pip install ultralytics` 後再啟動連續任務。"
            ) from exc

        self.device = resolve_yolo_device(device)
        self.conf = float(conf)
        self.yolo_interval = max(1, int(yolo_interval))
        self.side_cam_key = side_cam_key
        self.wrist_cam_key = wrist_cam_key
        self.grasp_stable_checks = max(1, int(grasp_stable_checks))
        self.put_stable_checks = max(1, int(put_stable_checks))
        self.drop_confirm_checks = max(1, int(drop_confirm_checks))
        self.lift_px = float(lift_px)
        self.cup_rim_pad_x = float(cup_rim_pad_x)
        self.cup_rim_height = float(cup_rim_height)
        self.gripper_open_thresh = float(gripper_open_thresh)
        self.gripper_open_delta = float(gripper_open_delta)
        self.yolo_mode = yolo_mode

        prompt_cfg = load_yoloe_prompt_config(prompt_config_path)
        self.names: list[str] = list(prompt_cfg.get("names", DEFAULT_NAMES))

        if yolo_mode == "yoloe":
            ref_image = prompt_cfg["ref_image"]
            boxes = prompt_cfg["boxes"]
            if not Path(ref_image).is_file():
                raise FileNotFoundError(
                    f"找不到 YOLOE 參考圖 {ref_image!r}。請先執行："
                    "`uv run python -u local/yolo_monitor.py --calibrate-ref --cam-side-45 2`"
                )
            self.model = YOLOE(weights_path)
            # Pre-encode visual prompt embeddings once before control loop starts
            self.model.predict(
                ref_image,
                refer_image=ref_image,
                verbose=False,
                device=self.device,
                visual_prompts=dict(bboxes=np.array(boxes), cls=np.arange(len(self.names))),
            )
        else:
            self.model = YOLO(weights_path)

        # Runtime state
        self.baseline_boxes: dict[str, DetectionBox] = {}
        self.cup_drop_roi: list[float] | None = None
        self.latest_detections: dict[str, DetectionBox] = {}
        self.latest_wrist_detections: dict[str, DetectionBox] = {}
        self._last_infer_step = -999
        self._last_infer_ms = 0.0

        # Per-stage latch/stability counters
        self.grasp_stable_count = 0
        self.put_stable_count = 0
        self.drop_confirm_count = 0
        self.entered_cup_zone = False
        self.gripper_released = False
        self._stage_start_gripper: float | None = None
        self._stage_start_shoulder: float | None = None

    def reset_stage_state(self, joint_state: np.ndarray | list[float] | None = None) -> None:
        """Reset latches and counters when entering or retrying a stage."""
        self.grasp_stable_count = 0
        self.put_stable_count = 0
        self.drop_confirm_count = 0
        self.entered_cup_zone = False
        self.gripper_released = False
        self._stage_start_gripper = float(joint_state[5]) if joint_state is not None and len(joint_state) >= 6 else None
        self._stage_start_shoulder = float(joint_state[1]) if joint_state is not None and len(joint_state) >= 2 else None

    def _extract_bgr(self, obs: dict[str, Any], cam_key: str) -> np.ndarray | None:
        if cam_key not in obs:
            return None
        img = np.asarray(obs[cam_key])
        if img.ndim != 3:
            return None
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _predict_frame(self, bgr: np.ndarray) -> dict[str, DetectionBox]:
        t0 = time.perf_counter()
        res = self.model.predict(bgr, conf=0.05, device=self.device, verbose=False)[0]
        self._last_infer_ms = (time.perf_counter() - t0) * 1000.0

        best: dict[str, DetectionBox] = {}
        if res.boxes is None or len(res.boxes) == 0:
            return best
        for score, cls_idx, box in zip(
            res.boxes.conf.tolist(), res.boxes.cls.tolist(), res.boxes.xyxy.tolist()
        ):
            idx = int(cls_idx)
            if idx < 0 or idx >= len(self.names) or score < self.conf:
                continue
            name = self.names[idx]
            prev = best.get(name)
            if prev is None or score > prev.conf:
                best[name] = DetectionBox(cls_name=name, bbox=[float(v) for v in box], conf=float(score))
        return best

    def update_detections(self, obs: dict[str, Any], step: int, force: bool = False) -> bool:
        """Run YOLOE inference when ``step`` hits ``yolo_interval`` (or ``force=True``)."""
        if not force and (step - self._last_infer_step) < self.yolo_interval:
            return False
        bgr_side = self._extract_bgr(obs, self.side_cam_key)
        if bgr_side is not None:
            self.latest_detections = self._predict_frame(bgr_side)
        bgr_wrist = self._extract_bgr(obs, self.wrist_cam_key)
        if bgr_wrist is not None:
            self.latest_wrist_detections = self._predict_frame(bgr_wrist)
        self._last_infer_step = step
        return True

    def capture_baseline(self, obs: dict[str, Any]) -> dict[str, DetectionBox]:
        """Snapshot desktop object positions while arm is at Home pose."""
        self.update_detections(obs, step=0, force=True)
        self._last_infer_step = -999
        for name, det in self.latest_detections.items():
            self.baseline_boxes[name] = det
        if "cup" in self.baseline_boxes:
            cx1, cy1, cx2, cy2 = self.baseline_boxes["cup"].bbox
            self.cup_drop_roi = [
                max(0.0, cx1 - self.cup_rim_pad_x),
                max(0.0, cy1 - self.cup_rim_height),
                cx2 + self.cup_rim_pad_x,
                cy2,
            ]
        return dict(self.baseline_boxes)

    @staticmethod
    def _extract_joint_state(obs: dict[str, Any]) -> np.ndarray | None:
        if "observation.state" in obs:
            arr = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
            return arr
        motor_order = (
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        )
        if all(k in obs for k in motor_order):
            return np.array([float(obs[k]) for k in motor_order], dtype=np.float32)
        return None

    def _is_lifted_pose(self, joint_state: np.ndarray | None) -> bool:
        if joint_state is None or len(joint_state) < 6:
            return True
        # During pick lift, shoulder_lift moves upward from table surface relative to stage start
        if self._stage_start_shoulder is None:
            return True
        return abs(float(joint_state[1]) - self._stage_start_shoulder) >= 5.0 or float(joint_state[5]) < 35.0

    def verify_grasp(
        self,
        target_name: str,
        obs: dict[str, Any],
        step: int,
        min_pick_steps: int = 90,
    ) -> StageVerificationStatus:
        """Check whether ``target_name`` was lifted off the table or captured in gripper."""
        ran_infer = self.update_detections(obs, step)
        joint_state = self._extract_joint_state(obs)
        if self._stage_start_gripper is None and joint_state is not None:
            self.reset_stage_state(joint_state)

        status = StageVerificationStatus(
            required_stable=self.grasp_stable_checks,
            detections=dict(self.latest_detections),
        )

        if step < min_pick_steps:
            status.reason = f"approach/grasp ({step}/{min_pick_steps} steps)"
            status.stable_count = self.grasp_stable_count
            return status

        base_det = self.baseline_boxes.get(target_name)
        curr_det = self.latest_detections.get(target_name)
        wrist_det = self.latest_wrist_detections.get(target_name)

        elevated = False
        occluded_from_baseline = False
        reason = "searching target"

        if curr_det is not None and base_det is not None:
            lift_delta = base_det.center[1] - curr_det.center[1]
            if lift_delta >= self.lift_px:
                elevated = True
                reason = f"lifted +{lift_delta:.0f}px"
            else:
                reason = f"on table (dy={lift_delta:+.0f}px)"
        elif curr_det is None and base_det is not None:
            # Small cube/teabag is often occluded by closed gripper jaws when lifted
            if wrist_det is not None:
                elevated = True
                reason = f"in wrist ROI ({wrist_det.conf:.2f})"
            elif self._is_lifted_pose(joint_state):
                occluded_from_baseline = True
                reason = "vanished from table baseline (held/occluded)"
        elif curr_det is not None and base_det is None and wrist_det is not None:
            elevated = True
            reason = f"detected in wrist ROI ({wrist_det.conf:.2f})"

        if ran_infer:
            if elevated or occluded_from_baseline:
                self.grasp_stable_count += 1
            else:
                self.grasp_stable_count = max(0, self.grasp_stable_count - 1)

        status.stable_count = self.grasp_stable_count
        status.reason = reason
        status.verified = self.grasp_stable_count >= self.grasp_stable_checks
        return status

    def verify_put_in_opaque_cup(
        self,
        target_name: str,
        obs: dict[str, Any],
        step: int,
        min_put_steps: int = 60,
    ) -> StageVerificationStatus:
        """Verify placement into an opaque cup using 3-phase latch + drop detection."""
        ran_infer = self.update_detections(obs, step)
        joint_state = self._extract_joint_state(obs)
        if self._stage_start_gripper is None and joint_state is not None:
            self.reset_stage_state(joint_state)

        curr_det = self.latest_detections.get(target_name)
        cup_det = self.latest_detections.get("cup") or self.baseline_boxes.get("cup")
        drop_roi = self.cup_drop_roi
        if cup_det is not None:
            cx1, cy1, cx2, cy2 = cup_det.bbox
            drop_roi = [
                max(0.0, cx1 - self.cup_rim_pad_x),
                max(0.0, cy1 - self.cup_rim_height),
                cx2 + self.cup_rim_pad_x,
                cy2,
            ]
            self.cup_drop_roi = drop_roi

        # Phase 1: Latch cup drop-zone entry
        if drop_roi is not None and curr_det is not None:
            if _point_in_roi(curr_det.center, drop_roi) or _boxes_overlap(curr_det.bbox, drop_roi):
                self.entered_cup_zone = True
        elif step >= min_put_steps and curr_det is None and drop_roi is not None:
            # Target remained occluded inside gripper while arm moved over cup
            self.entered_cup_zone = True

        # Phase 2: Latch gripper release from joint_state
        if step >= min_put_steps and joint_state is not None and len(joint_state) >= 6:
            grip_val = float(joint_state[5])
            opened_abs = grip_val >= self.gripper_open_thresh
            opened_rel = (
                self._stage_start_gripper is not None
                and (grip_val - self._stage_start_gripper) >= self.gripper_open_delta
            )
            if opened_abs or opened_rel:
                self.gripper_released = True

        # Check mid-flight or missed-cup drop outside cup on table
        dropped_outside = False
        if curr_det is not None and cup_det is not None:
            in_cup_or_rim = (
                (drop_roi is not None and (_point_in_roi(curr_det.center, drop_roi) or _boxes_overlap(curr_det.bbox, drop_roi)))
                or _boxes_overlap(curr_det.bbox, cup_det.bbox)
            )
            base_det = self.baseline_boxes.get(target_name)
            near_table_height = (
                base_det is None or abs(curr_det.center[1] - base_det.center[1]) < 45.0
            )
            if not in_cup_or_rim and near_table_height and (self.gripper_released or step >= min_put_steps):
                dropped_outside = True

        if ran_infer:
            if dropped_outside:
                self.drop_confirm_count += 1
                self.put_stable_count = 0
            else:
                self.drop_confirm_count = max(0, self.drop_confirm_count - 1)
                if self.entered_cup_zone and self.gripper_released and curr_det is None:
                    self.put_stable_count += 1
                elif curr_det is not None and not dropped_outside:
                    # Still visible above cup rim before dropping inside
                    self.put_stable_count = 0

        if self.drop_confirm_count >= self.drop_confirm_checks:
            reason = f"dropped outside cup ({self.drop_confirm_count}/{self.drop_confirm_checks})"
        elif self.entered_cup_zone and self.gripper_released and curr_det is None:
            reason = f"inside opaque cup (stable {self.put_stable_count}/{self.put_stable_checks})"
        elif self.entered_cup_zone and not self.gripper_released:
            reason = "above cup rim, waiting gripper release"
        else:
            reason = "moving object toward cup"

        return StageVerificationStatus(
            verified=self.put_stable_count >= self.put_stable_checks,
            dropped_on_table=self.drop_confirm_count >= self.drop_confirm_checks,
            stable_count=self.put_stable_count,
            required_stable=self.put_stable_checks,
            entered_cup_zone=self.entered_cup_zone,
            gripper_released=self.gripper_released,
            reason=reason,
            detections=dict(self.latest_detections),
        )

    def draw_hud(
        self,
        obs: dict[str, Any],
        stage_idx: int,
        total_stages: int,
        stage_label: str,
        switch_ms: float,
        elapsed_s: float,
        timeout_s: float,
        retries: int,
        status: StageVerificationStatus | None,
    ) -> np.ndarray | None:
        """Render annotated side_45 + wrist HUD frame in BGR format for cv2.imshow."""
        side_bgr = self._extract_bgr(obs, self.side_cam_key)
        if side_bgr is None:
            return None

        # Draw baseline boxes (dashed gray)
        for name, base_det in self.baseline_boxes.items():
            x1, y1, x2, y2 = map(int, base_det.bbox)
            cv2.rectangle(side_bgr, (x1, y1), (x2, y2), (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(side_bgr, f"init:{name}", (x1, min(side_bgr.shape[0] - 8, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Draw Cup Drop Zone ROI
        if self.cup_drop_roi is not None:
            rx1, ry1, rx2, ry2 = map(int, self.cup_drop_roi)
            cv2.rectangle(side_bgr, (rx1, ry1), (rx2, ry2), (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(side_bgr, "Cup Drop Zone", (rx1, max(16, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        # Draw current YOLOE detections
        for name, det in self.latest_detections.items():
            x1, y1, x2, y2 = map(int, det.bbox)
            color = DEFAULT_COLORS.get(name, (255, 255, 255))
            cv2.rectangle(side_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(side_bgr, f"{name} {det.conf:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        wrist_bgr = self._extract_bgr(obs, self.wrist_cam_key)
        if wrist_bgr is not None:
            for name, det in self.latest_wrist_detections.items():
                x1, y1, x2, y2 = map(int, det.bbox)
                color = DEFAULT_COLORS.get(name, (255, 255, 255))
                cv2.rectangle(wrist_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(wrist_bgr, f"{name} {det.conf:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if wrist_bgr.shape[0] != side_bgr.shape[0]:
                scale = side_bgr.shape[0] / wrist_bgr.shape[0]
                wrist_bgr = cv2.resize(wrist_bgr, (int(wrist_bgr.shape[1] * scale), side_bgr.shape[0]))
            canvas = np.hstack([wrist_bgr, side_bgr])
        else:
            canvas = side_bgr

        # Top translucent HUD banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 74), (20, 20, 20), -1)
        canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0)

        rem_s = max(0.0, timeout_s - elapsed_s)
        cv2.putText(
            canvas,
            f"Stage {stage_idx + 1}/{total_stages}: {stage_label} | Retry {retries} | Switch {switch_ms:.2f}ms | YOLOE {self._last_infer_ms:.1f}ms",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 160),
            2,
            cv2.LINE_AA,
        )
        if status is not None:
            prog = f"Stable {status.stable_count}/{status.required_stable}"
            cv2.putText(
                canvas,
                f"Status: {status.reason} | {prog} | Timeout in {rem_s:.1f}s | [n]ext [r]eset [q]uit",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        return canvas


class MockYOLOEPerceptionMonitor:
    """Deterministic perception simulator for offline FSM dry-run verification."""

    def __init__(
        self,
        grasp_stable_checks: int = 5,
        put_stable_checks: int = 8,
        simulate_failures: bool = False,
    ) -> None:
        self.grasp_stable_checks = grasp_stable_checks
        self.put_stable_checks = put_stable_checks
        self.simulate_failures = simulate_failures
        self.baseline_boxes: dict[str, DetectionBox] = {
            "cube": DetectionBox("cube", [240.0, 320.0, 290.0, 370.0], 0.88),
            "tea bag": DetectionBox("tea bag", [360.0, 330.0, 420.0, 380.0], 0.84),
            "cup": DetectionBox("cup", [470.0, 210.0, 560.0, 365.0], 0.92),
        }
        self._attempt_counter: dict[str, int] = {}
        self._stable_count = 0

    def reset_stage_state(self, joint_state: Any = None) -> None:
        self._stable_count = 0

    def capture_baseline(self, obs: dict[str, Any]) -> dict[str, DetectionBox]:
        return dict(self.baseline_boxes)

    def verify_grasp(
        self,
        target_name: str,
        obs: dict[str, Any],
        step: int,
        min_pick_steps: int = 10,
    ) -> StageVerificationStatus:
        key = f"pick:{target_name}"
        if step == 0:
            self._attempt_counter[key] = self._attempt_counter.get(key, 0) + 1
        attempt = self._attempt_counter.get(key, 1)

        # When simulating failures, fail the first pick attempt of cube by timing out
        if self.simulate_failures and target_name == "cube" and attempt == 1:
            return StageVerificationStatus(
                verified=False,
                stable_count=0,
                required_stable=self.grasp_stable_checks,
                reason="[SIM] missed grasp -> waiting timeout",
            )

        if step < min_pick_steps:
            return StageVerificationStatus(
                verified=False,
                stable_count=0,
                required_stable=self.grasp_stable_checks,
                reason=f"[SIM] approaching {target_name} ({step}/{min_pick_steps})",
            )

        self._stable_count += 1
        return StageVerificationStatus(
            verified=self._stable_count >= self.grasp_stable_checks,
            stable_count=self._stable_count,
            required_stable=self.grasp_stable_checks,
            reason=f"[SIM] {target_name} lifted +55px",
        )

    def verify_put_in_opaque_cup(
        self,
        target_name: str,
        obs: dict[str, Any],
        step: int,
        min_put_steps: int = 8,
    ) -> StageVerificationStatus:
        key = f"put:{target_name}"
        if step == 0:
            self._attempt_counter[key] = self._attempt_counter.get(key, 0) + 1
        attempt = self._attempt_counter.get(key, 1)

        # When simulating failures, drop the cube mid-flight on the first put attempt
        if self.simulate_failures and target_name == "cube" and attempt == 1 and step >= 6:
            return StageVerificationStatus(
                verified=False,
                dropped_on_table=True,
                stable_count=0,
                required_stable=self.put_stable_checks,
                reason="[SIM] cube dropped outside cup on table!",
            )

        if step < min_put_steps:
            return StageVerificationStatus(
                verified=False,
                entered_cup_zone=step >= min_put_steps // 2,
                gripper_released=False,
                stable_count=0,
                required_stable=self.put_stable_checks,
                reason=f"[SIM] carrying {target_name} to cup ({step}/{min_put_steps})",
            )

        self._stable_count += 1
        return StageVerificationStatus(
            verified=self._stable_count >= self.put_stable_checks,
            entered_cup_zone=True,
            gripper_released=True,
            stable_count=self._stable_count,
            required_stable=self.put_stable_checks,
            reason=f"[SIM] {target_name} inside opaque cup",
        )

    def draw_hud(self, *args: Any, **kwargs: Any) -> None:
        return None


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOE visual-prompt calibration and standalone preview.")
    parser.add_argument("--calibrate-ref", action="store_true", help="Capture reference frame and select 3 ROIs.")
    parser.add_argument("--cam-side-45", default="2", help="OpenCV camera index or path for side_45.")
    parser.add_argument("--image", default=None, help="Optional existing image path instead of opening camera.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Output/input JSON prompt config path.")
    parser.add_argument("--ref-out", default=str(DEFAULT_REF_IMAGE_PATH), help="Output path for captured reference jpg.")
    parser.add_argument("--weights", default="yoloe-11l-seg.pt", help="YOLOE weights path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--display", action="store_true", help="Run live YOLOE preview on side_45 camera.")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()
    if args.calibrate_ref:
        calibrate_yoloe_reference(
            cam_source=args.cam_side_45,
            image_path=args.image,
            out_config=args.config,
            out_ref_image=args.ref_out,
        )
        if not args.display:
            return

    if args.display:
        monitor = YOLOEPerceptionMonitor(
            weights_path=args.weights,
            prompt_config_path=args.config,
            conf=args.conf,
            yolo_interval=1,
        )
        src = int(args.cam_side_45) if str(args.cam_side_45).isdigit() else str(args.cam_side_45)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"無法開啟相機 {args.cam_side_45!r}")
        try:
            step = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                obs = {"side_45": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)}
                if step == 0:
                    monitor.capture_baseline(obs)
                monitor.update_detections(obs, step, force=True)
                hud = monitor.draw_hud(
                    obs=obs,
                    stage_idx=0,
                    total_stages=1,
                    stage_label="YOLOE Monitor Preview",
                    switch_ms=0.0,
                    elapsed_s=0.0,
                    timeout_s=99.0,
                    retries=0,
                    status=None,
                )
                if hud is not None:
                    cv2.imshow("YOLOE Perception Preview", hud)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    monitor.capture_baseline(obs)
                    print("Updated baseline boxes:", list(monitor.baseline_boxes.keys()))
                step += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
