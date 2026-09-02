"""
Dataloader for the GAN module.

Loads MNIST or Fashion-MNIST via torchvision, normalized to [-1, 1] to match
the Generator's Tanh output. GANs are trained purely on unlabeled real images
(the discriminator just needs to see "real" examples), so we don't bother
with a train/test split here — everything goes into one DataLoader.
"""

from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_DATASETS = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
}


def get_dataloader(
    batch_size: int = 64,
    dataset_name: str = "mnist",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> DataLoader:
    dataset_name = dataset_name.lower()
    if dataset_name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. Choose from {list(_DATASETS)}."
        )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> pixel range [-1, 1]
        ]
    )

    dataset_cls = _DATASETS[dataset_name]
    dataset = dataset_cls(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # keep it simple/portable, especially on Windows
    )
    return dataloader
