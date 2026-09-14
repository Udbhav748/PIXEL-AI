"""Small shared MNIST classifier used only for evaluating VAE/GAN outputs.

This is an evaluation-only helper -- separate from the vae/ and gan/ modules,
which stay independent per the project's design. It turns "does this look
right" into an actual number: how often a simple digit classifier recognizes
the model's output as a real, correct digit.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_CHECKPOINT = ROOT_DIR / "models" / "eval_classifier.pt"


class SimpleClassifier(nn.Module):
    """A small CNN digit classifier -- not part of the project's core modules,
    just a yardstick for evaluation. Two conv layers is enough for ~98%+
    accuracy on MNIST in a single epoch."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14x14 -> 7x7
        x = x.flatten(1)
        return self.fc(x)


def get_classifier(device, data_dir, epochs=1, batch_size=128):
    """Loads the classifier from models/eval_classifier.pt if it exists,
    otherwise trains it once (fast: ~1 epoch is enough) and saves it there,
    so repeated notebook runs don't retrain it every time."""
    model = SimpleClassifier().to(device)

    if CLASSIFIER_CHECKPOINT.exists():
        model.load_state_dict(torch.load(CLASSIFIER_CHECKPOINT, map_location=device))
        model.eval()
        return model

    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    CLASSIFIER_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_CHECKPOINT)
    return model


def accuracy(model, images, labels, device):
    """images: [N,1,28,28] in [0,1]. Returns accuracy as a float 0-1."""
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu()
    return (preds == labels).float().mean().item()


def predict(model, images, device):
    """images: [N,1,28,28] in [0,1]. Returns (predicted_labels, confidence),
    both length-N tensors on CPU."""
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(images.to(device)), dim=1)
        conf, preds = probs.max(dim=1)
    return preds.cpu(), conf.cpu()
