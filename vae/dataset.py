"""
MNIST dataloaders for the VAE.

Images are normalized to [0, 1] only (just ToTensor — no mean/std standardization)
since the decoder ends in a sigmoid and expects pixel values in that range.
"""

from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size: int = 128, data_dir: str | Path = "data"):
    """
    Downloads MNIST (if not already present) into data_dir and returns
    (train_loader, test_loader).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()  # converts PIL image [0,255] -> tensor [0,1]

    train_dataset = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
