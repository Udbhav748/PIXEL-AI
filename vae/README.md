# VAE Module

This module trains a Variational Autoencoder (VAE) on MNIST and provides an
inference interface for the Streamlit app.

## What's a VAE?

A regular autoencoder squishes an image down into a small vector (the "latent
code") and learns to rebuild the image from it. A VAE does something similar
but with a twist: instead of encoding an image into one fixed vector, it
encodes it into a *distribution* — a mean (`mu`) and a variance (`logvar`) —
and then samples a point `z` from that distribution before decoding.

The pipeline looks like this:

```
image -> encoder -> (mu, logvar) -> reparameterization -> z -> decoder -> reconstruction
```

The "reparameterization trick" is how we sample `z` in a way that still lets
gradients flow back through the network during training: instead of sampling
`z` directly (which isn't differentiable), we compute

```
z = mu + eps * std          where eps ~ N(0, 1) and std = exp(0.5 * logvar)
```

Because sampling `eps` from a distribution independent of the network doesn't
block gradients from reaching `mu` and `logvar`, we can backpropagate normally.

## Loss function

Training uses two loss terms added together:

- **Reconstruction loss** (binary cross-entropy) — how close is the
  reconstructed image to the original? This pushes the model to actually
  rebuild the input correctly.
- **KL divergence** — how far is the learned distribution `N(mu, sigma^2)`
  from a standard normal `N(0, 1)`? This keeps the latent space organized and
  "gap-free," which is what makes random sampling from `N(0, 1)` later
  actually produce reasonable-looking digits instead of noise.

`total_loss = reconstruction_loss + kl_loss`

## About the dataset (important limitation)

This VAE is trained **only on MNIST** — 28x28 grayscale handwritten digits,
nothing else. It will not meaningfully reconstruct arbitrary photos uploaded
by a user; if you feed it a photo of a cat or a face, it'll just squash it
down to grayscale 28x28 and output something that vaguely looks like a blurry
digit, because that's literally the only kind of image it has ever learned to
produce. The Streamlit UI should make this limitation clear rather than
implying the model does general-purpose image reconstruction.

## Files

- `model.py` — the `VAE` class (encoder, reparameterize, decoder) and `vae_loss()`.
- `dataset.py` — `get_dataloaders()`, downloads/loads MNIST.
- `train.py` — training script, saves checkpoint + result plots.
- `inference.py` — `load_model()`, `reconstruct()`, `generate()` — used by the Streamlit app.

## How to train

From the project root:

```
python vae/train.py --epochs 15
```

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--latent-dim`,
`--max-batches` (caps batches per epoch, handy for a quick smoke test),
`--data-dir`, `--out-dir`, `--results-dir`.

Quick smoke test (fast, doesn't actually train anything useful):

```
python vae/train.py --epochs 1 --max-batches 5
```

This saves:
- `models/vae/vae.pt` — the checkpoint
- `results/vae/loss_curve.png` — training loss over epochs
- `results/vae/reconstructions.png` — original vs. reconstructed test images
- `results/vae/samples.png` — images decoded from random latent vectors

## How to run inference standalone

```
python vae/inference.py
```

This loads the checkpoint from `models/vae/vae.pt` (training must be run
first) and does a quick reconstruct + generate smoke test, printing whether
it succeeded.
