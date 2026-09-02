"""
Train a basic (vanilla) GAN on MNIST or Fashion-MNIST.

Adversarial training in a nutshell: the Generator tries to produce images
that look real, the Discriminator tries to tell real images apart from the
Generator's fakes. They're trained alternately, each step against the
other's current behavior — that's why it's called "adversarial." As
training progresses, the generator's loss and discriminator's loss don't
necessarily go down together like in normal supervised training; a healthy
GAN often keeps both losses bouncing around similar values instead of
converging to zero. If the generator's loss collapses very low, it often
means the discriminator got fooled too easily (or the generator found a
shortcut that produces only a few kinds of outputs, known as mode
collapse).

Usage:
    python gan/train.py --epochs 30
    python gan/train.py --epochs 1 --max-batches 5   # quick smoke test
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

# Make `gan` importable when this file is run directly (python gan/train.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gan.dataset import get_dataloader
from gan.model import Discriminator, Generator

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUT_DIR = ROOT_DIR / "models" / "gan"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "gan"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a vanilla GAN on MNIST/Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist"])
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap the number of batches per epoch (useful for smoke testing)",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="save a sample image grid every N epochs (default: every epoch)",
    )
    return parser.parse_args()


def save_sample_grid(generator, latent_dim, device, path, n=25):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        gen_imgs = generator(z)
    generator.train()
    save_image(gen_imgs, path, nrow=5, normalize=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    img_shape = (1, 28, 28)
    dataloader = get_dataloader(
        batch_size=args.batch_size, dataset_name=args.dataset, data_dir=args.data_dir
    )

    generator = Generator(latent_dim=args.latent_dim, img_shape=img_shape).to(device)
    discriminator = Discriminator(img_shape=img_shape).to(device)

    adversarial_loss = nn.BCELoss()
    # betas=(0.5, 0.999) is the standard choice for GAN training (from the
    # original DCGAN paper) — the lower beta1 makes Adam react faster to the
    # constantly-shifting adversarial objective than its usual default of 0.9.
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        g_loss_total, d_loss_total, n_batches = 0.0, 0.0, 0

        for i, (imgs, _) in enumerate(dataloader):
            if args.max_batches is not None and i >= args.max_batches:
                break

            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)

            # Ground-truth labels: 1 = real, 0 = fake
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            # The generator wants the discriminator to call its fakes "real",
            # so we compute its loss against the `valid` labels.
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)

            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # The discriminator sees both real images (should predict "valid")
            # and generated images (should predict "fake"). We detach gen_imgs
            # here so this step doesn't also backprop into the generator.
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            n_batches += 1

        avg_g_loss = g_loss_total / max(n_batches, 1)
        avg_d_loss = d_loss_total / max(n_batches, 1)
        print(f"[Epoch {epoch + 1}/{args.epochs}] D loss: {avg_d_loss:.4f}  G loss: {avg_g_loss:.4f}")

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            sample_path = args.results_dir / f"epoch_{epoch + 1:03d}.png"
            save_sample_grid(generator, args.latent_dim, device, sample_path)

    # Final sample grid + checkpoints
    save_sample_grid(generator, args.latent_dim, device, args.results_dir / "final_samples.png")

    torch.save(
        {
            "state_dict": generator.state_dict(),
            "latent_dim": args.latent_dim,
            "img_shape": img_shape,
        },
        args.out_dir / "generator.pt",
    )
    torch.save(
        {
            "state_dict": discriminator.state_dict(),
            "img_shape": img_shape,
        },
        args.out_dir / "discriminator.pt",
    )
    print(f"Saved checkpoints to {args.out_dir}")
    print(f"Saved sample images to {args.results_dir}")


if __name__ == "__main__":
    main()
