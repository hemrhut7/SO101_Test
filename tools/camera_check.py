import argparse
import sys
from pathlib import Path
import cv2


def _default_backend() -> int:
    """Use V4L2 on Linux to avoid FFMPEG false positives on metadata nodes."""
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _suppress_opencv_logs() -> None:
    """Suppress noisy OpenCV probe warnings when scanning nonexistent camera IDs."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass


def _get_device_name(cid: int) -> str | None:
    """Return the V4L2 device name on Linux if available."""
    sys_name = Path(f"/sys/class/video4linux/video{cid}/name")
    if sys_name.exists():
        return sys_name.read_text(encoding="utf-8", errors="ignore").strip()
    return None


def _collect_candidate_ids(max_index: int) -> list[int]:
    """Build a sorted list of candidate camera indices to probe."""
    candidates = set(range(max_index))
    v4l_dir = Path("/dev")
    if v4l_dir.exists():
        for dev in v4l_dir.glob("video*"):
            suffix = dev.name.removeprefix("video")
            if suffix.isdigit():
                candidates.add(int(suffix))
    return sorted(candidates)


def scan_cameras(max_index: int = 10) -> list[dict]:
    """Scan all accessible cameras and return their IDs and metadata."""
    _suppress_opencv_logs()
    found: list[dict] = []

    backend_flag = _default_backend()
    for cid in _collect_candidate_ids(max_index):
        cap = cv2.VideoCapture(cid, backend_flag)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue

        height, width = frame.shape[:2]
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
        found.append(
            {
                "id": cid,
                "name": _get_device_name(cid),
                "width": width,
                "height": height,
                "backend": backend,
            }
        )
        cap.release()

    return found


def open_cameras(
    cam_ids: list[int], width: int = 640, height: int = 480, fps: int = 30
) -> dict[int, cv2.VideoCapture]:
    """Open cameras with MJPG format to avoid USB bandwidth saturation."""
    caps: dict[int, cv2.VideoCapture] = {}
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    backend_flag = _default_backend()
    for cid in cam_ids:
        cap = cv2.VideoCapture(cid, backend_flag)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        caps[cid] = cap
    return caps


def preview_cameras(caps: dict[int, cv2.VideoCapture]) -> None:
    """Display live camera feeds with their IDs until 'q' or ESC is pressed."""
    print("開啟相機預覽視窗，按 'q' 或 ESC 離開...")
    try:
        while True:
            for cid, cap in caps.items():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.putText(
                        frame,
                        f"ID: {cid}",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(f"Camera {cid}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="掃描電腦上所有可用相機與其編號")
    parser.add_argument("--max-index", type=int, default=10, help="掃描的最大相機編號上限 (預設: 10)")
    parser.add_argument("--scan-only", action="store_true", help="只掃描並顯示相機編號，不開啟預覽視窗")
    parser.add_argument("--width", type=int, default=640, help="預覽寬度 (預設: 640)")
    parser.add_argument("--height", type=int, default=480, help="預覽高度 (預設: 480)")
    parser.add_argument("--fps", type=int, default=30, help="預覽幀率 (預設: 30)")
    args = parser.parse_args()

    cameras = scan_cameras(max_index=args.max_index)
    if not cameras:
        print("找不到任何可用相機。")
        return

    print(f"共找到 {len(cameras)} 支可用相機：")
    for cam in cameras:
        name_str = f" | 名稱: {cam['name']}" if cam["name"] else ""
        print(
            f"  - 編號 {cam['id']}: {cam['width']}x{cam['height']} "
            f"(後端: {cam['backend']}{name_str})"
        )

    cam_ids = [cam["id"] for cam in cameras]
    print(f"可用相機編號列表: {cam_ids}")

    if args.scan_only:
        return

    caps = open_cameras(cam_ids, width=args.width, height=args.height, fps=args.fps)
    if not caps:
        print("無法開啟任何相機進行預覽。")
        return
    preview_cameras(caps)


if __name__ == "__main__":
    main()
