from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch import nn


class TinyVisuomotorPolicy(nn.Module):
    def __init__(self, num_actions: int, state_dim: int = 2) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(96 + 32, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(96, num_actions),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_feature = self.image_encoder(image)
        state_feature = self.state_encoder(state)
        return self.policy_head(torch.cat([image_feature, state_feature], dim=1))


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_frame(frame_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)
    return tensor


def normalized_state(cursor_x: float, cursor_y: float, device: torch.device) -> torch.Tensor:
    state = np.array([[cursor_x * 2.0 - 1.0, cursor_y * 2.0 - 1.0]], dtype=np.float32)
    return torch.from_numpy(state).to(device)


def load_policy_checkpoint(path: Path, device: torch.device) -> Tuple[TinyVisuomotorPolicy, Dict]:
    checkpoint = torch.load(path, map_location=device)
    labels = checkpoint["label_names"]
    model = TinyVisuomotorPolicy(num_actions=len(labels), state_dim=checkpoint.get("state_dim", 2))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint

