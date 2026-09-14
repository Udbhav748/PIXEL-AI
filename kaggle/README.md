# Retraining on Kaggle (GPU)

This project was originally trained on CPU, which meant modest epoch
budgets and (as documented in `gan/README.md`) some interrupted training
runs. `train_on_kaggle.ipynb` here retrains the VAE and both GAN
architectures on a Kaggle GPU instead, with:

- Much higher epoch budgets (GPU is roughly 10-20x faster than the CPU
  runs this project was trained with).
- **DCGAN-paper-style weight initialization** (`weights_init()` in
  `gan/train.py`) — tight `N(0, 0.02)` init for conv/linear layers instead
  of PyTorch's default, a standard GAN stabilizer.
- **TTUR** (two-timescale update rule) — the discriminator's learning rate
  (`--lr-d`) set lower than the generator's, so it doesn't outpace the
  generator the way our CPU runs showed it tending to.
- **Larger VAE latent space** (32 instead of the CPU-trained checkpoint's
  20) — more capacity to preserve digit identity through the encode/decode
  round-trip.

## How to use it

1. Upload `train_on_kaggle.ipynb` to [kaggle.com](https://www.kaggle.com/code) (New Notebook → upload).
2. In the right-hand panel: **Settings → Accelerator → GPU P100** (our
   training scripts are single-GPU, so P100 beats "GPU T4 x2" here — the
   second T4 would sit idle).
3. **Run All.**
4. Once it finishes, download `pixel_ai_retrained.zip` from the **Output**
   tab.
5. Unzip it and copy its `models/`, `results/`, and `notebooks/*.ipynb`
   into your local `PIXEL-AI/` folder, overwriting the existing ones.
6. `git add -A && git commit -m "..." && git push`.
7. Re-check the numbers quoted in the root `README.md` (PSNR, accuracy,
   AUC, class coverage) against what the re-executed notebooks print now,
   and update the text if they moved — they were written against the
   CPU-trained checkpoints.

`--lr`, `--lr-d`, `--epochs` etc. are exposed as plain variables in the
notebook's config cell if you want to try different values.
