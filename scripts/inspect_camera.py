from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lesson_common import ensure_dir, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe connected cameras and save one frame per available index.")
    parser.add_argument("--max-index", type=int, default=6, help="Highest camera index to test.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument("--backend", default="auto", choices=["auto", "avfoundation", "dshow", "v4l2"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/camera_probe"))
    return parser.parse_args()


def open_with_backend(index: int, backend: str) -> cv2.VideoCapture:
    backend_id = 0
    if backend == "avfoundation":
        backend_id = cv2.CAP_AVFOUNDATION
    elif backend == "dshow":
        backend_id = cv2.CAP_DSHOW
    elif backend == "v4l2":
        backend_id = cv2.CAP_V4L2
    return cv2.VideoCapture(index, backend_id) if backend_id else cv2.VideoCapture(index)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    print(f"Saving probe frames to: {out_dir.resolve()}")

    found = []
    for index in range(args.max_index + 1):
        cap = open_with_backend(index, args.backend)
        if not cap.isOpened():
            print(f"[{index}] not available")
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        for _ in range(5):
            cap.read()

        ok, frame = cap.read()
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if not ok or frame is None:
            print(f"[{index}] opened, but frame read failed")
            continue

        filename = out_dir / f"camera_{index}_{timestamp_string()}.jpg"
        cv2.imwrite(str(filename), frame)
        found.append(index)
        print(f"[{index}] OK  requested={args.width}x{args.height}@{args.fps}  actual={actual_width}x{actual_height}@{actual_fps:.1f}  file={filename}")

    if not found:
        print("No camera produced a valid frame. Check USB connection and camera permissions.")
    else:
        print(f"Available camera indices: {found}")


if __name__ == "__main__":
    main()

