"""
Inference helpers for the VAE module. This is what the Streamlit app imports.

Function names/signatures are a fixed contract — do not rename:
    load_model(checkpoint_path, device=None) -> VAE
    reconstruct(model, image: PIL.Image.Image) -> PIL.Image.Image
    generate(model, n=8) -> list[PIL.Image.Image]

Note: this VAE is trained only on MNIST (28x28 grayscale handwritten digits).
Reconstructing an arbitrary photo will just produce a blurry digit-like blob,
not a meaningful reconstruction of that photo — the UI calling this should
make that limitation clear to the user rather than implying general-purpose
image reconstruction.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Support being run both as a script (`python vae/inference.py`) and as a
# package module (`from vae.inference import load_model`, e.g. from Streamlit).
try:
    from .model import VAE
except ImportError:
    from model import VAE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "vae" / "vae.pt"

# Preprocessing: grayscale + resize to the 28x28 the model was trained on, then
# to a [0, 1] tensor (matches the ToTensor-only normalization used in training).
_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def load_model(checkpoint_path: Path | str = DEFAULT_CHECKPOINT, device: str | None = None) -> VAE:
    """Loads a trained VAE checkpoint and returns it in eval mode on the right device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VAE(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def _tensor_to_pil(tensor: torch.Tensor, upscale_to: int = 128) -> Image.Image:
    """Converts a single-image [1, 28, 28] tensor in [0,1] to an upscaled PIL image."""
    array = (tensor.squeeze().clamp(0, 1).numpy() * 255).astype("uint8")
    image = Image.fromarray(array, mode="L")
    image = image.resize((upscale_to, upscale_to), resample=Image.NEAREST)
    return image


def reconstruct(model: VAE, image: Image.Image) -> Image.Image:
    """
    Runs one uploaded image through the VAE and returns the reconstruction.

    The image is converted to grayscale and resized to 28x28 (what the model
    expects) before being encoded/decoded. The result is upscaled back up for
    display, but it's still fundamentally a 28x28-resolution reconstruction —
    don't expect fine detail.
    """
    device = next(model.parameters()).device

    x = _preprocess(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    with torch.no_grad():
        recon, _mu, _logvar = model(x)

    return _tensor_to_pil(recon.cpu()[0])


def generate(model: VAE, n: int = 8) -> list[Image.Image]:
    """Samples n random latent vectors from N(0,1) and decodes them into images."""
    device = next(model.parameters()).device

    with torch.no_grad():
        z = torch.randn(n, model.latent_dim).to(device)
        samples = model.decode(z).cpu()

    return [_tensor_to_pil(samples[i]) for i in range(n)]


if __name__ == "__main__":
    if not DEFAULT_CHECKPOINT.exists():
        print(f"No checkpoint found at {DEFAULT_CHECKPOINT}.")
        print("Train the model first: python vae/train.py")
    else:
        model = load_model(DEFAULT_CHECKPOINT)
        print(f"Loaded checkpoint from {DEFAULT_CHECKPOINT}")

        # smoke test: reconstruct a blank-ish image and generate a few samples
        dummy_image = Image.new("L", (64, 64), color=128)
        recon_image = reconstruct(model, dummy_image)
        print(f"reconstruct() OK, output image size: {recon_image.size}")

        samples = generate(model, n=4)
        print(f"generate() OK, produced {len(samples)} images, each size: {samples[0].size}")

        print("Inference smoke test passed.")
