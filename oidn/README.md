# OIDN module

Intel Open Image Denoise (OIDN) is a pretrained denoising filter, originally
built to clean up noisy ray-traced renders. It is not a generative model —
it only removes noise from an existing image, it can't invent new images or
reconstruct missing content the way the VAE or GAN modules do.

## What denoising means here

A noisy image is a clean image plus random pixel-level noise (from a low
light sensor, high ISO, compression, or in our case synthetic Gaussian
noise). Denoising tries to recover something close to the clean image
without blurring away real detail.

## How OIDN works (at a high level)

OIDN ships a small pretrained CNN, compiled directly into its runtime
libraries (see `models/oidn/README.md`). Given a color image, the filter
predicts a cleaned-up version in a single forward pass — no training or
per-image optimization needed on our side. It normally also accepts albedo
and normal auxiliary buffers (useful info from a renderer) for better
results; this project only uses the color-only "RT" mode since we don't
have render buffers for ordinary photos.

## Input / output

- Input: an RGB image (PNG/JPG/etc.), any resolution.
- Output: an RGB image of the same size, with noise reduced.

## Limitations

- Tuned for rendered images, not natural photos — results on heavily
  compressed or very noisy real-world photos can be inconsistent.
- No parameters to tune besides HDR on/off; it's not trainable from this
  project.
- Won't hallucinate detail that isn't there, unlike a generative model.

## Files

- `inference.py` — `denoise()` runs OIDN on a PIL image; `add_synthetic_noise()`
  is a helper for demoing on images that aren't already noisy.
- `sample_images/` — a bundled noisy/clean pair for quick testing.

## Try it standalone

```bash
python oidn/inference.py
```

Saves a denoised copy of the bundled sample image to `results/oidn/`.
