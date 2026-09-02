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

## Experiments & findings

We ran into mode collapse — generated digits clustering around a couple of repeated shapes instead of covering all 10 — and tried two things to address it.

| Attempt | Setup | D loss (final) | G loss (final) | Sample diversity |
|---|---|---|---|---|
| 1. Baseline | 25 epochs, no label smoothing | ~0.21 | ~2.8 | Low — mostly 2s and 3s |
| 2. Train longer | 75 epochs, no label smoothing | 0.13 (still falling) | 3.8 (still rising) | Still low — discriminator just kept overpowering the generator |
| 3. Label smoothing | ~25 epochs, real-label target 0.9 instead of 1.0 | 0.28-0.30 (plateaued) | 2.8-2.9 (plateaued) | Still low — similar clustering to attempt 1 |

**Attempt 1 → 2 (train longer):** made things worse, not better. The discriminator kept getting stronger the whole time (D loss trending toward 0), meaning the generator's feedback signal kept getting weaker. More epochs alone doesn't fix mode collapse in a GAN — sometimes it entrenches it, since the imbalance has more time to run away.

**Attempt 2 → 3 (label smoothing):** trained the discriminator to predict `0.9` instead of a full `1.0` for real images (`--label-smoothing`, default `0.9`), so it can't get overconfident. This worked exactly as intended — both losses stopped drifting and settled into a stable range instead. But sample diversity barely changed.

**Diagnosis:** label smoothing fixes *training stability* (an overconfident discriminator), which is a real problem, but it isn't the same problem as mode collapse (the generator settling for a narrow set of outputs). They often show up together, which makes it tempting to assume one fix handles both — this project's result shows that isn't guaranteed. Actually fixing the diversity problem would need something that directly penalizes the generator for producing similar outputs within a batch (minibatch discrimination, feature matching) or a fundamentally better-suited architecture (a convolutional DCGAN instead of this MLP GAN). Both are reasonable next steps but out of scope for a first "basic GAN" implementation — noted here rather than silently left out.

The pre-label-smoothing checkpoint (attempt 1, 25 epochs) is kept at `models/gan/backup_25ep/` for comparison, alongside the current label-smoothed one at `models/gan/generator.pt`.

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
