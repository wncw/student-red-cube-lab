from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


ACTION_LABELS = ["left", "right", "up", "down", "close"]

LABEL_KEYS = {
    ord("a"): "left",
    ord("d"): "right",
    ord("w"): "up",
    ord("s"): "down",
    32: "close",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    backend: str = "auto",
) -> cv2.VideoCapture:
    backend_id = 0
    if backend == "avfoundation":
        backend_id = cv2.CAP_AVFOUNDATION
    elif backend == "dshow":
        backend_id = cv2.CAP_DSHOW
    elif backend == "v4l2":
        backend_id = cv2.CAP_V4L2

    cap = cv2.VideoCapture(camera_index, backend_id) if backend_id else cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {camera_index} could not be opened.")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Camera index {camera_index} opened, but no frame was received.")

    return cap


def camera_info(cap: cv2.VideoCapture) -> Dict[str, float]:
    return {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }


def resize_for_display(frame: np.ndarray, display_width: int) -> np.ndarray:
    if display_width <= 0 or frame.shape[1] <= display_width:
        return frame
    scale = display_width / float(frame.shape[1])
    display_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)


def draw_text_panel(
    frame: np.ndarray,
    lines: Iterable[str],
    origin: Tuple[int, int] = (16, 28),
    line_height: int = 24,
) -> None:
    lines = list(lines)
    if not lines:
        return

    x, y = origin
    width = max(260, max(len(line) for line in lines) * 10 + 24)
    height = line_height * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 24), (x - 8 + width, y - 24 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_crosshair(frame: np.ndarray, x: int, y: int, color: Tuple[int, int, int]) -> None:
    size = 18
    cv2.line(frame, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 6, color, 2, cv2.LINE_AA)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

