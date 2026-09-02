"""
Trains the VAE on MNIST.

Usage:
    python vae/train.py --epochs 15
    python vae/train.py --epochs 1 --max-batches 5   # quick smoke test

Run from the project root so relative default paths resolve correctly.
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from model import VAE, vae_loss
from dataset import get_dataloaders

# Project root = parent of this file's folder (vae/), used for default paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=None,
                         help="cap number of batches per epoch, for quick smoke testing")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "models" / "vae"),
                         help="where to save the checkpoint")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results" / "vae"),
                         help="where to save plots / sample images")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, max_batches=None):
    model.train()
    total_loss_sum, recon_loss_sum, kl_loss_sum = 0.0, 0.0, 0.0
    n_batches = 0

    for batch_idx, (images, _labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        n_batches += 1

    return total_loss_sum / n_batches, recon_loss_sum / n_batches, kl_loss_sum / n_batches


def save_loss_curve(history, out_path: Path):
    epochs = range(1, len(history["total"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["total"], label="total loss")
    plt.plot(epochs, history["recon"], label="reconstruction loss")
    plt.plot(epochs, history["kl"], label="KL loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (per image)")
    plt.title("VAE training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_reconstructions(model, test_loader, device, out_path: Path, n=8):
    model.eval()
    images, _labels = next(iter(test_loader))
    images = images[:n].to(device)

    with torch.no_grad():
        recon, _mu, _logvar = model(images)

    # stack originals on top row, reconstructions on bottom row
    comparison = torch.cat([images.cpu(), recon.cpu()])
    grid = make_grid(comparison, nrow=n)

    plt.figure(figsize=(n, 2.5))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("Top: original | Bottom: reconstruction")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_random_samples(model, device, latent_dim, out_path: Path, n=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        samples = model.decode(z).cpu()

    grid = make_grid(samples, nrow=4)
    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("Random samples from latent space")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, data_dir=args.data_dir)

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"total": [], "recon": [], "kl": []}

    for epoch in range(1, args.epochs + 1):
        total, recon, kl = train_one_epoch(model, train_loader, optimizer, device, max_batches=args.max_batches)
        history["total"].append(total)
        history["recon"].append(recon)
        history["kl"].append(kl)
        print(f"Epoch {epoch}/{args.epochs} - total: {total:.4f}  recon: {recon:.4f}  kl: {kl:.4f}")

    checkpoint_path = out_dir / "vae.pt"
    torch.save({"state_dict": model.state_dict(), "latent_dim": args.latent_dim}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    save_loss_curve(history, results_dir / "loss_curve.png")
    save_reconstructions(model, test_loader, device, results_dir / "reconstructions.png")
    save_random_samples(model, device, args.latent_dim, results_dir / "samples.png")
    print(f"Saved result images to {results_dir}")


if __name__ == "__main__":
    main()
