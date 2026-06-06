from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from lesson_common import ACTION_LABELS, ensure_dir
from tiny_policy import TinyVisuomotorPolicy, bgr_frame_to_chw, select_device


class RedCubeActionDataset(Dataset):
    def __init__(self, dataset_dir: Path, image_size: int, augment: bool) -> None:
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.augment = augment
        self.rows = self._load_rows(dataset_dir / "labels.csv")
        self.label_to_index = {label: idx for idx, label in enumerate(ACTION_LABELS)}

    def _load_rows(self, labels_path: Path) -> List[Dict[str, str]]:
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found: {labels_path}")
        with labels_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        valid_rows = []
        for row in rows:
            image_path = self.dataset_dir / row["image"]
            if row.get("label") in ACTION_LABELS and image_path.exists():
                valid_rows.append(row)
        if not valid_rows:
            raise RuntimeError("No valid samples found. Collect data first.")
        return valid_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = self.dataset_dir / row["image"]
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            frame = self._augment(frame)

        image = torch.from_numpy(bgr_frame_to_chw(frame, self.image_size)).float()

        cursor_x = float(row["cursor_x"])
        cursor_y = float(row["cursor_y"])
        state = torch.tensor([cursor_x * 2.0 - 1.0, cursor_y * 2.0 - 1.0], dtype=torch.float32)
        label = torch.tensor(self.label_to_index[row["label"]], dtype=torch.long)
        return image, state, label

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12.0, 12.0)
        augmented = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        if random.random() < 0.25:
            noise = np.random.normal(0, 4, augmented.shape).astype(np.int16)
            augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return augmented

    def label_counts(self) -> Counter:
        return Counter(row["label"] for row in self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny end-to-end imitation policy from camera images and virtual gripper state.")
    parser.add_argument("--dataset", type=Path, default=Path("data/e2e_red_cube"))
    parser.add_argument("--model-out", type=Path, default=Path("models/e2e_red_cube_policy.pt"))
    parser.add_argument("--summary-out", type=Path, default=Path("models/e2e_red_cube_training_summary.json"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, states, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        states = states.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, states)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, List[List[int]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    matrix = [[0 for _ in ACTION_LABELS] for _ in ACTION_LABELS]
    for images, states, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        states = states.to(device)
        labels = labels.to(device)

        logits = model(images, states)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * labels.size(0)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        for target, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            matrix[target][pred] += 1
    return total_loss / max(1, total), correct / max(1, total), matrix


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = RedCubeActionDataset(args.dataset, image_size=args.image_size, augment=not args.no_augment)
    counts = dataset.label_counts()
    print(f"Samples: {len(dataset)}")
    print("Label counts:", dict(counts))
    for label in ACTION_LABELS:
        if counts[label] < 10:
            print(f"Warning: label '{label}' has only {counts[label]} samples. Add more data for stable behavior.")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device()
    print(f"Device: {device}")
    model = TinyVisuomotorPolicy(num_actions=len(ACTION_LABELS), state_dim=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = -1.0
    best_state = None
    best_matrix = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, matrix = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_matrix = matrix

    ensure_dir(args.model_out.parent)
    checkpoint = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "label_names": ACTION_LABELS,
        "state_dim": 2,
        "image_size": args.image_size,
        "class_counts": dict(counts),
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, args.model_out)
    print(f"Saved model: {args.model_out.resolve()}")

    summary = {
        "dataset": str(args.dataset),
        "model_out": str(args.model_out),
        "num_samples": len(dataset),
        "class_counts": dict(counts),
        "history": history,
        "best_val_acc": best_val_acc,
        "confusion_matrix_labels": ACTION_LABELS,
        "confusion_matrix": best_matrix,
    }
    ensure_dir(args.summary_out.parent)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved summary: {args.summary_out.resolve()}")


if __name__ == "__main__":
    main()
