from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from lesson_common import draw_crosshair, draw_text_panel, ensure_dir, open_camera, resize_for_display, timestamp_string


DEFAULT_THRESHOLDS = {
    "h1_low": 0,
    "h1_high": 10,
    "h2_low": 170,
    "h2_high": 180,
    "s_low": 80,
    "v_low": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a red cube with OpenCV HSV thresholding.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=1920, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument("--backend", default="auto", choices=["auto", "avfoundation", "dshow", "v4l2"])
    parser.add_argument("--min-area", type=int, default=800, help="Ignore red contours smaller than this area.")
    parser.add_argument("--display-width", type=int, default=1280, help="Resize window for display only.")
    parser.add_argument("--show-mask", action="store_true", help="Show threshold mask next to camera frame.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/direct_tracker"))
    return parser.parse_args()


def make_red_mask(frame: np.ndarray, thresholds: Dict[str, int]) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([thresholds["h1_low"], thresholds["s_low"], thresholds["v_low"]], dtype=np.uint8)
    upper1 = np.array([thresholds["h1_high"], 255, 255], dtype=np.uint8)
    lower2 = np.array([thresholds["h2_low"], thresholds["s_low"], thresholds["v_low"]], dtype=np.uint8)
    upper2 = np.array([thresholds["h2_high"], 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def largest_red_square(mask: np.ndarray, min_area: int) -> Optional[Tuple[int, int, int, int, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2
    sx = max(0, cx - side // 2)
    sy = max(0, cy - side // 2)
    return sx, sy, side, side, area


def create_tuning_window(thresholds: Dict[str, int]) -> None:
    cv2.namedWindow("HSV tuning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV tuning", 420, 260)
    cv2.createTrackbar("H1 low", "HSV tuning", thresholds["h1_low"], 180, lambda _: None)
    cv2.createTrackbar("H1 high", "HSV tuning", thresholds["h1_high"], 180, lambda _: None)
    cv2.createTrackbar("H2 low", "HSV tuning", thresholds["h2_low"], 180, lambda _: None)
    cv2.createTrackbar("H2 high", "HSV tuning", thresholds["h2_high"], 180, lambda _: None)
    cv2.createTrackbar("S low", "HSV tuning", thresholds["s_low"], 255, lambda _: None)
    cv2.createTrackbar("V low", "HSV tuning", thresholds["v_low"], 255, lambda _: None)


def read_tuning_values() -> Dict[str, int]:
    return {
        "h1_low": cv2.getTrackbarPos("H1 low", "HSV tuning"),
        "h1_high": cv2.getTrackbarPos("H1 high", "HSV tuning"),
        "h2_low": cv2.getTrackbarPos("H2 low", "HSV tuning"),
        "h2_high": cv2.getTrackbarPos("H2 high", "HSV tuning"),
        "s_low": cv2.getTrackbarPos("S low", "HSV tuning"),
        "v_low": cv2.getTrackbarPos("V low", "HSV tuning"),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    cap = open_camera(args.camera, args.width, args.height, args.fps, args.backend)
    thresholds = dict(DEFAULT_THRESHOLDS)
    show_mask = args.show_mask
    tuning = False
    window_name = "Direct OpenCV red cube tracker"

    print("Controls: q/ESC quit, s screenshot, t HSV tuning, m show/hide mask")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed.")
                break

            if tuning:
                thresholds = read_tuning_values()

            mask = make_red_mask(frame, thresholds)
            result = largest_red_square(mask, args.min_area)

            display = frame.copy()
            if result is None:
                status = "not detected"
                draw_text_panel(display, ["Direct OpenCV tracker", "red cube: not detected", "press t to tune HSV"])
            else:
                x, y, w, h, area = result
                cx = x + w // 2
                cy = y + h // 2
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
                draw_crosshair(display, cx, cy, (0, 255, 255))
                cv2.putText(display, f"center=({cx}, {cy}) area={area:.0f}", (x, max(28, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                status = f"center=({cx}, {cy}) area={area:.0f}"
                draw_text_panel(
                    display,
                    [
                        "Direct OpenCV tracker",
                        f"red cube: {status}",
                        f"HSV red: H {thresholds['h1_low']}-{thresholds['h1_high']} or {thresholds['h2_low']}-{thresholds['h2_high']}",
                        f"S>={thresholds['s_low']} V>={thresholds['v_low']}",
                    ],
                )

            if show_mask:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_bgr, "mask", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                display = np.hstack([display, mask_bgr])

            display = resize_for_display(display, args.display_width)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = args.out_dir / f"direct_tracker_{timestamp_string()}.jpg"
                cv2.imwrite(str(path), display)
                print(f"Saved screenshot: {path}")
            if key == ord("m"):
                show_mask = not show_mask
            if key == ord("t"):
                tuning = not tuning
                if tuning:
                    create_tuning_window(thresholds)
                else:
                    cv2.destroyWindow("HSV tuning")
                print(f"HSV tuning: {'on' if tuning else 'off'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

