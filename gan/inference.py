"""
Inference helpers for the GAN module — this is what the Streamlit app
imports.

    from gan.inference import load_generator, generate

    model = load_generator("models/gan/generator.pt")
    images = generate(model, n=8)   # list of PIL.Image.Image

Note: this GAN only generates brand-new images from random noise. It does
not take an input image, denoise it, or reconstruct it — there's no "input"
at all besides a random latent vector. That's different from OIDN (which
denoises a given noisy image) and the VAE (which encodes/reconstructs a
given input image). If you want "restore this specific image," look there.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gan.model import Generator

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("Pillow is required for gan.inference (pip install pillow)") from e

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "gan" / "generator.pt"

DISPLAY_SIZE = 128  # upscale generated images for easier viewing in the UI


def load_generator(checkpoint_path: Path | str, device: str | None = None) -> Generator:
    """Load a trained Generator from a checkpoint saved by gan/train.py."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = Generator(
        latent_dim=checkpoint["latent_dim"],
        img_shape=tuple(checkpoint["img_shape"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def generate(model: Generator, n: int = 8) -> list:
    """Sample `n` random latent vectors and return a list of PIL images.

    Generator output is in [-1, 1] (Tanh), so we rescale to [0, 255] uint8
    before building PIL images, and resize up so they're easy to see.
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, device=device)
        gen_imgs = model(z)  # (n, C, H, W), range [-1, 1]

    # [-1, 1] -> [0, 1] -> [0, 255]
    gen_imgs = (gen_imgs.clamp(-1, 1) + 1) / 2
    gen_imgs = (gen_imgs * 255).to(torch.uint8).cpu()

    images = []
    for img_tensor in gen_imgs:
        img_array = img_tensor.squeeze(0).numpy()  # drop channel dim (grayscale)
        pil_img = Image.fromarray(img_array, mode="L")
        pil_img = pil_img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
        images.append(pil_img)

    return images


if __name__ == "__main__":
    if not DEFAULT_CHECKPOINT.exists():
        print(
            f"No checkpoint found at {DEFAULT_CHECKPOINT}.\n"
            "Train the model first, e.g.:\n"
            "    python gan/train.py --epochs 30"
        )
    else:
        model = load_generator(DEFAULT_CHECKPOINT)
        images = generate(model, n=4)
        print(f"Loaded checkpoint from {DEFAULT_CHECKPOINT}")
        print(f"Generated {len(images)} images, size {images[0].size}, mode {images[0].mode}")
        print("Inference smoke test passed.")
