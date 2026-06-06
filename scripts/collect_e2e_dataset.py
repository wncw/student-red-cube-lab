from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Dict

import cv2

from lesson_common import (
    ACTION_LABELS,
    LABEL_KEYS,
    clamp01,
    draw_virtual_gripper,
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
    parser.add_argument("--append", action="store_true", help="Append samples to an existing dataset instead of resetting it.")
    parser.add_argument("--reset", action="store_true", help=argparse.SUPPRESS)
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
            image = row.get("image")
            image_exists = image is not None and (labels_path.parent / image).exists()
            if label in counts and image_exists:
                counts[label] += 1
    return counts


def labels_has_rows(labels_path: Path) -> bool:
    if not labels_path.exists():
        return False
    with labels_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return any(True for _ in reader)


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
    if args.reset and args.append:
        raise RuntimeError("Use either --append or --reset, not both.")

    cap = open_camera(args.camera, args.width, args.height, args.fps, args.backend)
    if args.dataset.exists() and not args.append:
        shutil.rmtree(args.dataset)

    collection_mode = "append existing dataset" if args.append else "new dataset (reset)"
    dataset_dir = ensure_dir(args.dataset)
    images_dir = ensure_dir(dataset_dir / "images")
    labels_path = dataset_dir / "labels.csv"
    metadata_path = dataset_dir / "metadata.json"

    write_json(
        metadata_path,
        {
            "created_or_updated_at": timestamp_string(),
            "collection_mode": collection_mode,
            "task": "red cube virtual-gripper action labeling",
            "observation": {
                "image": [args.save_height, args.save_width, 3],
                "state": ["cursor_x_normalized", "cursor_y_normalized"],
            },
            "action_labels": ACTION_LABELS,
            "note": "Images include the virtual gripper marker. Cursor state is also stored separately.",
        },
    )

    sample_index = next_sample_index(images_dir)
    if sample_index == 0 and labels_has_rows(labels_path):
        cap.release()
        raise RuntimeError(
            "labels.csv exists but no sample images were found. "
            "This usually means only the images were deleted. "
            "Run this script without --append to reset, or delete the whole dataset directory."
        )
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
    print(f"Mode: {collection_mode}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed.")
                break

            display = frame.copy()
            draw_virtual_gripper(display, cursor_x, cursor_y)

            count_line = " ".join([f"{label}:{counts[label]}" for label in ACTION_LABELS])
            draw_text_panel(
                display,
                [
                    "End-to-End dataset collection",
                    f"Mode: {collection_mode}",
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
                saved_frame = frame.copy()
                draw_virtual_gripper(saved_frame, cursor_x, cursor_y)
                saved = cv2.resize(saved_frame, (args.save_width, args.save_height), interpolation=cv2.INTER_AREA)
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
