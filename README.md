# PIXEL AI — Image Restoration & Generative AI Lab

A small classroom project exploring three different approaches to "AI + images":
denoising with a pretrained filter (OIDN), reconstruction/generation with a
Variational Autoencoder (VAE), and pure generation with a GAN. All three are
kept as separate, independent modules — they solve different problems and
aren't meant to be combined into one network — with a single Streamlit app
tying them together.

## Project overview

- **OIDN** — takes a noisy image, removes the noise. Pretrained, no training
  involved on our end.
- **VAE** — learns to encode MNIST digits into a compact latent space and
  decode them back; can reconstruct a digit or generate a new one from a
  random point in that latent space.
- **GAN** — learns to generate new MNIST digits from random noise, by
  training a generator against a discriminator.

These solve genuinely different problems and shouldn't be confused for one
another: OIDN restores an existing image, the VAE gives you a structured
latent space plus reconstruction/generation, and the GAN only generates from
noise with no input image at all.

## Architecture

**OIDN:**
```
Noisy Image → OIDN → Clean Image
```

**VAE:**
```
Image → Encoder → (mu, logvar) → Reparameterize → z → Decoder → Reconstruction
```

**GAN:**
```
Noise → Generator → Fake Image
                        ↑
        Discriminator ←-┘  (also sees real images)
```

## Project structure

```
PIXEL-AI/
├── app/streamlit_app.py     Streamlit frontend, ties all three modules together
├── oidn/                    OIDN inference (pretrained, no training script)
├── vae/                     VAE model, dataset, train, inference
├── gan/                     GAN model, dataset, train, inference
├── notebooks/                01_OIDN.ipynb, 02_VAE.ipynb, 03_GAN.ipynb
├── models/                   saved checkpoints (oidn/ has none — see its README)
├── results/                   saved plots and sample images from training
└── data/                       downloaded datasets (MNIST etc., gitignored)
```

Each module folder (`oidn/`, `vae/`, `gan/`) has its own README with the
concepts, architecture, and commands specific to that module.

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Then:
```bash
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app/streamlit_app.py
```

The OIDN tab works immediately (no training needed). The VAE and GAN tabs
need a trained checkpoint first — see Training below. If a checkpoint is
missing, the app tells you so rather than training silently in the
background.

## Training

```bash
python vae/train.py --epochs 15
python gan/train.py --epochs 30
```

Both scripts run on CPU or GPU automatically (`cuda` if available, else
`cpu`) and expose `--epochs`, `--batch-size`, `--lr`, and `--latent-dim`,
among other flags — see each module's README for the full list. A quick
smoke test (`--epochs 1 --max-batches 5`) is useful for checking everything
runs before committing to a full training run.

## Results

- `results/vae/loss_curve.png`, `reconstructions.png`, `samples.png`
- `results/gan/epoch_XXX.png` (samples across training), `final_samples.png`
- `results/oidn/sample_denoised.png`

## Model comparison

| Model | Purpose                     | Input                  | Output                            |
|-------|------------------------------|-------------------------|-------------------------------------|
| OIDN  | Denoising                    | Noisy image             | Clean image                         |
| VAE   | Reconstruction / generation  | Image / latent vector   | Reconstruction / generated image    |
| GAN   | Generation                   | Random noise            | Synthetic image                     |

## Future experiments

- **Data augmentation with GAN samples** — train a small classifier on
  MNIST with vs. without GAN-generated samples mixed in, and compare
  accuracy. Not implemented here, just a natural next step given the GAN
  module already exists.
- **Pix2Pix** — image-to-image translation is a different (paired,
  conditional) setup from the unconditional GAN here. Interesting future
  direction, not part of this project.

## References

These are the exact resources provided for this assignment:

- [PyTorch-GAN — basic GAN implementation](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py) — primary reference for the GAN module (generator/discriminator architecture, training loop, loss).
- [GANs for Data Augmentation and Privacy (Coursera assignment)](https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Apply%20Generative%20Adversarial%20Networks%20(GANs)/Week%201%20-%20GANs%20for%20Data%20Augmentation%20and%20Privacy/C3W1_Assignment.ipynb) — reference for the GAN/data-augmentation workflow.
- [VAE notebook (Coursera assignment)](https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Build%20Better%20Generative%20Adversarial%20Networks%20(GANs)/Week%202%20-%20GAN%20Disadvantages%20and%20Bias/C2W2_VAE.ipynb) — primary reference for the VAE module (encoder/decoder, reparameterization, loss).
- [Pix2Pix (Coursera assignment)](https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Apply%20Generative%20Adversarial%20Networks%20(GANs)/Week%202%20-%20Image-to-Image%20Translation%20with%20Pix2Pix/C3W2A_Assignment.ipynb) — reference only, not implemented (see Future experiments).

## Limitations

- The VAE and GAN are both trained on MNIST only — they work with 28x28
  grayscale digits, not arbitrary photos. The Streamlit UI makes this clear
  rather than pretending otherwise.
- Both are deliberately simple architectures (a small conv VAE, an MLP GAN)
  chosen for readability over state-of-the-art image quality.
- OIDN is tuned for ray-traced renders, not general photography — results
  on real-world noisy photos can be inconsistent.
