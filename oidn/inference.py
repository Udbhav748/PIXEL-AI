"""OIDN inference: run Intel Open Image Denoise on an image.

OIDN was built for cleaning up noisy ray-traced renders, not general
photos, but it works reasonably well on ordinary noisy images too when
used in color-only mode (no albedo/normal auxiliary buffers).
"""

from pathlib import Path

import numpy as np
from PIL import Image

import pyoidn

BASE_DIR = Path(__file__).parent
SAMPLE_NOISY = BASE_DIR / "sample_images" / "sample_noisy.png"


def denoise(image: Image.Image) -> Image.Image:
    """Run OIDN's RT filter (color-only) on an RGB image and return the result."""
    rgb = image.convert("RGB")
    color = np.ascontiguousarray(np.array(rgb, dtype=np.float32) / 255.0)
    result = np.zeros_like(color, dtype=np.float32)

    device = pyoidn.Device()
    device.commit()

    flt = pyoidn.Filter(device, "RT")
    flt.set_image(pyoidn.OIDN_IMAGE_COLOR, color, pyoidn.OIDN_FORMAT_FLOAT3)
    flt.set_image(pyoidn.OIDN_IMAGE_OUTPUT, result, pyoidn.OIDN_FORMAT_FLOAT3)
    flt.set_bool("hdr", False)
    flt.commit()
    flt.execute()

    error = device.get_error()
    if error is not None:
        flt.release()
        device.release()
        raise RuntimeError(f"OIDN error: {error}")

    flt.release()
    device.release()

    result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result_u8)


def add_synthetic_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise to a clean image, useful when no real noisy photo is on hand."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


if __name__ == "__main__":
    if not SAMPLE_NOISY.exists():
        raise SystemExit(f"Sample image not found: {SAMPLE_NOISY}")

    img = Image.open(SAMPLE_NOISY)
    out = denoise(img)

    out_dir = BASE_DIR.parent / "results" / "oidn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_denoised.png"
    out.save(out_path)
    print(f"Denoised sample saved to {out_path}")
