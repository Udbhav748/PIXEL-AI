# GAN module

This module trains a basic GAN (Generative Adversarial Network) on MNIST (or Fashion-MNIST) to generate new digit images from random noise.

**Important:** this GAN only generates brand new images from random noise — it doesn't take an input image and denoise or reconstruct it. If you're looking for "clean up this noisy image," that's OIDN. If you're looking for "encode and reconstruct this specific image," that's the VAE. The GAN here has no input image at all, just a random vector.

## How it works

A GAN is two networks trained against each other:

- **Generator** — takes a random noise vector (100 numbers sampled from a normal distribution, the "latent vector") and maps it through a few fully connected layers into a 28x28 image.
- **Discriminator** — takes an image (real from the dataset, or fake from the generator) and outputs a single number: the probability it thinks the image is real.

They're trained in alternation, one batch at a time:

1. The generator makes a batch of fake images, and we update it to make the discriminator's output on those fakes closer to "real" (1.0). This is the generator "trying to fool" the discriminator.
2. The discriminator then sees a batch of real images (labeled 1) and a batch of fake images (labeled 0), and we update it to get better at telling them apart.

Both use binary cross-entropy (BCE) loss, since it's just a real-vs-fake classification problem at each step.

### What the losses mean

- **D loss** (discriminator loss) — how badly the discriminator is doing at distinguishing real from fake. If this goes to ~0, the discriminator has gotten too good and the generator probably isn't learning much from it anymore.
- **G loss** (generator loss) — how badly the generator is doing at fooling the discriminator. If this goes to ~0, the discriminator got fooled too easily, which often means something's wrong (or the generator found a cheap trick).

Unlike normal supervised training, you don't want either loss to just steadily decrease to zero — a healthy GAN usually has both losses oscillating around similar values, because the two networks keep adapting to each other. It's a moving target.

### Common problems

- **Mode collapse** — the generator finds a small set of outputs that reliably fool the discriminator and just keeps producing those, instead of the full variety of digits. You'd notice this as generated samples that all look suspiciously similar.
- **Training instability** — since the generator and discriminator are chasing a moving target, loss curves can be noisy/oscillating rather than smoothly decreasing. This is normal to an extent, but if one network overpowers the other early on, training can stall.

This project uses the simplest possible GAN (MLP layers only, no convolutions) to keep the architecture easy to follow. It's more prone to instability and mode collapse than fancier variants (DCGAN, WGAN, etc.), but that's a fair tradeoff for a first GAN implementation.

**Label smoothing:** training the discriminator on real images target `0.9` instead of a full `1.0` (`--label-smoothing`, default `0.9`) keeps it from getting overconfident too early — an overconfident discriminator gives the generator a weak, uninformative gradient, which is one of the things that pushes a GAN toward mode collapse. We tried just training longer first (75 epochs vs. 25) and it didn't help — the discriminator only got stronger, not weaker — so label smoothing was added instead.

**Result:** label smoothing did what it's supposed to do — D loss and G loss both plateaued instead of drifting further apart (D loss stayed around 0.27-0.30 instead of dropping to 0.13 like the un-smoothed 75-epoch run). But sample diversity didn't meaningfully improve — generated digits still cluster around a few shapes. That's a useful, honest result for a course project: it shows label smoothing fixes the *training stability* symptom, but mode collapse in a plain MLP GAN is a deeper architectural limitation that needs a structurally different fix (minibatch discrimination, a conv-based DCGAN, or similar) — out of scope for this basic implementation.

## Files

- `model.py` — `Generator` and `Discriminator` classes.
- `dataset.py` — `get_dataloader()`, loads MNIST/Fashion-MNIST normalized to [-1, 1].
- `train.py` — training loop, saves checkpoints and sample image grids.
- `inference.py` — `load_generator()` / `generate()`, used by the Streamlit app.

## Training

```
python gan/train.py --epochs 30
```

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--latent-dim`, `--dataset {mnist,fashion-mnist}`, `--max-batches` (cap batches/epoch, handy for a quick test run), `--data-dir`, `--out-dir`, `--results-dir`, `--sample-every`.

Quick smoke test (fast, doesn't actually train anything meaningful):

```
python gan/train.py --epochs 1 --max-batches 5
```

This saves:
- Generator/discriminator checkpoints to `models/gan/generator.pt` and `models/gan/discriminator.pt`.
- A sample image grid every `--sample-every` epochs to `results/gan/epoch_XXX.png`, plus a `results/gan/final_samples.png` at the end — useful for watching how sample quality changes over training.

## Running inference standalone

```
python gan/inference.py
```

Loads `models/gan/generator.pt` and generates a few sample images, printing a success message. If no checkpoint exists yet, it prints a message telling you to train first instead of training automatically.
