from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from lesson_common import ACTION_LABELS, clamp01, draw_crosshair, draw_text_panel, open_camera, resize_for_display
from tiny_policy import load_policy_checkpoint, normalized_state, preprocess_frame, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tiny end-to-end imitation policy on live camera frames.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/e2e_red_cube_policy.pt"))
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument("--backend", default="auto", choices=["auto", "avfoundation", "dshow", "v4l2"])
    parser.add_argument("--display-width", type=int, default=1280, help="Resize window for display only.")
    parser.add_argument("--cursor-step", type=float, default=0.035, help="Manual and auto cursor step in normalized coordinates.")
    parser.add_argument("--auto-cursor", action="store_true", help="Move virtual cursor using predicted actions.")
    return parser.parse_args()


def apply_action_to_cursor(label: str, cursor_x: float, cursor_y: float, step: float) -> tuple[float, float]:
    if label == "left":
        cursor_x -= step
    elif label == "right":
        cursor_x += step
    elif label == "up":
        cursor_y -= step
    elif label == "down":
        cursor_y += step
    return clamp01(cursor_x), clamp01(cursor_y)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = select_device()
    model, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    label_names = checkpoint.get("label_names", ACTION_LABELS)
    image_size = int(checkpoint.get("image_size", 96))
    cap = open_camera(args.camera, args.width, args.height, args.fps, args.backend)

    cursor_x = 0.5
    cursor_y = 0.5
    auto_cursor = args.auto_cursor
    predicted_label = "none"
    confidence = 0.0
    window_name = "End-to-End imitation policy"

    print("Controls: j/l/i/k move cursor, c center, p auto cursor, q/ESC quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed.")
                break

            with torch.no_grad():
                image = preprocess_frame(frame, image_size=image_size, device=device)
                state = normalized_state(cursor_x, cursor_y, device=device)
                logits = model(image, state)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu()
                pred_idx = int(torch.argmax(probs).item())
                predicted_label = label_names[pred_idx]
                confidence = float(probs[pred_idx].item())

            if auto_cursor and predicted_label != "close":
                cursor_x, cursor_y = apply_action_to_cursor(predicted_label, cursor_x, cursor_y, args.cursor_step)

            display = frame.copy()
            h, w = display.shape[:2]
            cursor_px = int(cursor_x * (w - 1))
            cursor_py = int(cursor_y * (h - 1))
            color = (0, 255, 0) if predicted_label == "close" else (255, 255, 0)
            draw_crosshair(display, cursor_px, cursor_py, color)
            cv2.rectangle(display, (cursor_px - 36, cursor_py - 36), (cursor_px + 36, cursor_py + 36), color, 2)

            sorted_probs = sorted(zip(label_names, probs.tolist()), key=lambda item: item[1], reverse=True)
            top_lines = [f"{label}: {prob:.2f}" for label, prob in sorted_probs[:3]]
            draw_text_panel(
                display,
                [
                    "End-to-End imitation policy",
                    f"predicted action: {predicted_label} ({confidence:.2f})",
                    f"cursor=({cursor_x:.2f}, {cursor_y:.2f}) auto={'on' if auto_cursor else 'off'}",
                    "No OpenCV red threshold is used here.",
                    *top_lines,
                ],
            )

            cv2.imshow(window_name, resize_for_display(display, args.display_width))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                cursor_x, cursor_y = 0.5, 0.5
            elif key == ord("j"):
                cursor_x = clamp01(cursor_x - args.cursor_step)
            elif key == ord("l"):
                cursor_x = clamp01(cursor_x + args.cursor_step)
            elif key == ord("i"):
                cursor_y = clamp01(cursor_y - args.cursor_step)
            elif key == ord("k"):
                cursor_y = clamp01(cursor_y + args.cursor_step)
            elif key == ord("p"):
                auto_cursor = not auto_cursor
                print(f"auto cursor: {'on' if auto_cursor else 'off'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

