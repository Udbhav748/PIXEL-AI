# GAN module

This module trains a GAN (Generative Adversarial Network) on MNIST (or Fashion-MNIST) to generate new digit images from random noise. Two architectures are implemented: the original basic **MLP** GAN, and a small **DCGAN** (convolutional) variant added later as a follow-up experiment — see Experiments & findings below for why, and how they compare.

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

| Attempt | Setup | D loss (final) | G loss (final) | Sample diversity | Sample sharpness |
|---|---|---|---|---|---|
| 1. Baseline | MLP, 25 epochs, no label smoothing | ~0.21 | ~2.8 | Low — mostly 2s and 3s | Speckled/noisy |
| 2. Train longer | MLP, 75 epochs, no label smoothing | 0.13 (still falling) | 3.8 (still rising) | Still low | Still speckled |
| 3. Label smoothing | MLP, ~25 epochs, real-label target 0.9 | 0.28-0.30 (plateaued) | 2.8-2.9 (plateaued) | Still low, similar to attempt 1 | Still speckled |
| 4. DCGAN | Conv G/D, 25 epochs, label smoothing kept | 0.25-0.30 (stable) | 2.6-3.3 (stable) | **Clearly better** — 0/2/3/5/7/8/9-like shapes visible in one grid | **Clearly better** — smooth strokes, no speckle noise |

**Attempt 1 → 2 (train longer):** made things worse, not better. The discriminator kept getting stronger the whole time (D loss trending toward 0), meaning the generator's feedback signal kept getting weaker. More epochs alone doesn't fix mode collapse in a GAN — sometimes it entrenches it, since the imbalance has more time to run away.

**Attempt 2 → 3 (label smoothing):** trained the discriminator to predict `0.9` instead of a full `1.0` for real images (`--label-smoothing`, default `0.9`), so it can't get overconfident. This worked exactly as intended — both losses stopped drifting and settled into a stable range instead. But sample diversity barely changed, and samples were still visibly grainy — an MLP has no notion that neighboring pixels should look similar, so speckle noise is a structural property of this architecture, not something a loss-function tweak fixes.

**Attempt 3 → 4 (DCGAN — swap MLP layers for convolutions):** kept label smoothing, but replaced both the generator and discriminator with small conv-based networks (`DCGenerator`/`DCDiscriminator` in `model.py` — see the "DCGAN variant" comment block there for why convolutions help). This is the one that actually delivered: noticeably smoother strokes (a direct result of convolution's built-in spatial locality) *and* noticeably more varied digit shapes in the same 25-epoch budget, even though nothing about the loss function or training length changed from attempt 3. It's not perfect — this is still a small, quickly-trained model, not a polished generator — but it's a clear, visible step up.

**Diagnosis:** this project ended up separating two failure modes that are easy to conflate. *Training instability* (an overconfident discriminator, diverging losses) is a loss-function/hyperparameter problem — label smoothing fixed it directly. *Mode collapse and blurriness* (limited output variety, speckled pixels) turned out to be closer to an *architecture* problem — the MLP's lack of spatial structure — and switching to convolutions fixed both symptoms together, better than any loss-side tweak did on its own.

Checkpoints for all the meaningfully different attempts are kept for comparison: `models/gan/backup_25ep/` (attempt 1, no label smoothing), `models/gan/generator.pt` (attempt 3, MLP + label smoothing — also the "MLP (basic)" option in the Streamlit app), and `models/gan/generator_dcgan.pt` (attempt 4, DCGAN — the "DCGAN (conv)" option in the app). Pick either one in the GAN tab to compare them side by side.

## Files

- `model.py` — `Generator`/`Discriminator` (MLP) and `DCGenerator`/`DCDiscriminator` (conv) classes.
- `dataset.py` — `get_dataloader()`, loads MNIST/Fashion-MNIST normalized to [-1, 1].
- `train.py` — training loop, saves checkpoints and sample image grids.
- `inference.py` — `load_generator()` / `generate()`, used by the Streamlit app. Auto-detects which architecture a checkpoint is (recorded in the checkpoint file) and rebuilds the right model class.

## Training

```
python gan/train.py --epochs 30                          # MLP (basic), the default
python gan/train.py --architecture dcgan --epochs 25      # DCGAN (conv)
```

Useful flags: `--architecture {mlp,dcgan}`, `--epochs`, `--batch-size`, `--lr` (generator learning rate), `--lr-d` (discriminator learning rate, defaults to `--lr` if not set — see TTUR note below), `--latent-dim`, `--dataset {mnist,fashion-mnist}`, `--label-smoothing` (default `0.9`), `--max-batches` (cap batches/epoch, handy for a quick test run), `--data-dir`, `--out-dir`, `--results-dir`, `--sample-every`, `--resume` (continue from the existing checkpoint for the chosen architecture instead of starting fresh — handy if a long run gets interrupted, since checkpoints are also saved periodically during training, not just at the end).

**Weight initialization:** both architectures now get the standard DCGAN-paper init (`weights_init()` in `train.py`) — conv/linear weights from a tight `N(0, 0.02)` instead of PyTorch's wider default, BatchNorm scale from `N(1, 0.02)`. Applied automatically on a fresh run (skipped when `--resume` actually finds a checkpoint, since then you want the resumed weights, not a reset).

**TTUR (`--lr-d`):** set the discriminator's learning rate lower than the generator's (e.g. `--lr 2e-4 --lr-d 1e-4`) to slow it down relative to the generator — directly targets the same "discriminator overpowers the generator" problem label smoothing partially addressed, from the optimizer side instead of the loss side. See `kaggle/` for a GPU training setup that uses this.

The two architectures save to separate files so training one never overwrites the other: `generator.pt`/`discriminator.pt` for MLP, `generator_dcgan.pt`/`discriminator_dcgan.pt` for DCGAN (same pattern for the `epoch_XXX*.png`/`final_samples*.png` result images).

Quick smoke test (fast, doesn't actually train anything meaningful):

```
python gan/train.py --epochs 1 --max-batches 5
```

This saves:
- Generator/discriminator checkpoints to `models/gan/`.
- A sample image grid every `--sample-every` epochs to `results/gan/epoch_XXX*.png`, plus a `final_samples*.png` at the end — useful for watching how sample quality changes over training.

## Running inference standalone

```
python gan/inference.py
```

Loads `models/gan/generator.pt` (the MLP checkpoint) and generates a few sample images, printing a success message. If no checkpoint exists yet, it prints a message telling you to train first instead of training automatically. To try the DCGAN checkpoint instead, use `gan.inference.load_generator(gan.inference.DCGAN_CHECKPOINT)` from a Python shell, or just pick "DCGAN (conv)" in the Streamlit app's GAN tab.
