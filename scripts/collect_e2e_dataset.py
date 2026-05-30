from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import cv2

from lesson_common import (
    ACTION_LABELS,
    LABEL_KEYS,
    clamp01,
    draw_crosshair,
    draw_text_panel,
    ensure_dir,
    open_camera,
    resize_for_display,
    timestamp_string,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect image + virtual gripper state -> action labels for a tiny end-to-end imitation policy.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--dataset", type=Path, default=Path("data/e2e_red_cube"), help="Dataset directory.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    parser.add_argument("--backend", default="auto", choices=["auto", "avfoundation", "dshow", "v4l2"])
    parser.add_argument("--save-width", type=int, default=320, help="Stored image width for training.")
    parser.add_argument("--save-height", type=int, default=240, help="Stored image height for training.")
    parser.add_argument("--display-width", type=int, default=1280, help="Resize window for display only.")
    parser.add_argument("--cursor-step", type=float, default=0.035, help="Virtual cursor move step in normalized coordinates.")
    return parser.parse_args()


def next_sample_index(images_dir: Path) -> int:
    max_index = -1
    for path in images_dir.glob("sample_*.jpg"):
        stem = path.stem.replace("sample_", "")
        if stem.isdigit():
            max_index = max(max_index, int(stem))
    return max_index + 1


def load_counts(labels_path: Path) -> Dict[str, int]:
    counts = {label: 0 for label in ACTION_LABELS}
    if not labels_path.exists():
        return counts

    with labels_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label")
            if label in counts:
                counts[label] += 1
    return counts


def append_row(labels_path: Path, row: Dict[str, object]) -> None:
    exists = labels_path.exists()
    fieldnames = ["image", "label", "cursor_x", "cursor_y", "width", "height", "timestamp"]
    with labels_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_dir = ensure_dir(args.dataset)
    images_dir = ensure_dir(dataset_dir / "images")
    labels_path = dataset_dir / "labels.csv"
    metadata_path = dataset_dir / "metadata.json"

    write_json(
        metadata_path,
        {
            "created_or_updated_at": timestamp_string(),
            "task": "red cube virtual-gripper action labeling",
            "observation": {
                "image": [args.save_height, args.save_width, 3],
                "state": ["cursor_x_normalized", "cursor_y_normalized"],
            },
            "action_labels": ACTION_LABELS,
            "note": "Images are saved without the virtual cursor overlay. Cursor state is stored separately.",
        },
    )

    cap = open_camera(args.camera, args.width, args.height, args.fps, args.backend)
    sample_index = next_sample_index(images_dir)
    counts = load_counts(labels_path)
    cursor_x = 0.5
    cursor_y = 0.5
    last_label = "none"
    window_name = "Collect E2E imitation dataset"

    print("Controls:")
    print("  labels: a=left d=right w=up s=down space=close")
    print("  cursor: j=left l=right i=up k=down c=center")
    print("  quit: q or ESC")
    print(f"Dataset: {dataset_dir.resolve()}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed.")
                break

            display = frame.copy()
            h, w = display.shape[:2]
            cursor_px = int(cursor_x * (w - 1))
            cursor_py = int(cursor_y * (h - 1))
            draw_crosshair(display, cursor_px, cursor_py, (255, 255, 0))
            cv2.rectangle(display, (cursor_px - 36, cursor_py - 36), (cursor_px + 36, cursor_py + 36), (255, 255, 0), 2)

            count_line = " ".join([f"{label}:{counts[label]}" for label in ACTION_LABELS])
            draw_text_panel(
                display,
                [
                    "End-to-End dataset collection",
                    "Label: a left | d right | w up | s down | space close",
                    "Cursor: j/l/i/k move | c center",
                    f"cursor=({cursor_x:.2f}, {cursor_y:.2f}) last={last_label}",
                    count_line,
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
            elif key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                saved = cv2.resize(frame, (args.save_width, args.save_height), interpolation=cv2.INTER_AREA)
                image_name = f"sample_{sample_index:06d}.jpg"
                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), saved, [cv2.IMWRITE_JPEG_QUALITY, 92])
                append_row(
                    labels_path,
                    {
                        "image": f"images/{image_name}",
                        "label": label,
                        "cursor_x": f"{cursor_x:.6f}",
                        "cursor_y": f"{cursor_y:.6f}",
                        "width": args.save_width,
                        "height": args.save_height,
                        "timestamp": f"{time.time():.6f}",
                    },
                )
                counts[label] += 1
                last_label = label
                print(f"[{sample_index:06d}] label={label} cursor=({cursor_x:.3f},{cursor_y:.3f}) image={image_name}")
                sample_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved labels: {labels_path.resolve()}")
        print(f"Final counts: {counts}")


if __name__ == "__main__":
    main()

