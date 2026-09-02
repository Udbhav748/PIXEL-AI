"""
VAE (Variational Autoencoder) model for MNIST digits.

Pipeline: image -> encoder -> (mu, logvar) -> reparameterize -> z -> decoder -> reconstruction

A normal autoencoder just learns to compress an image into a single latent vector
and decompress it back. A VAE instead learns a *distribution* (mean + variance)
over the latent space for each input. Sampling from that distribution during
training forces nearby points in latent space to decode to similar images, which
is what lets us generate new digits later by sampling random latent vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        # Small conv stack: 28x28x1 -> 14x14x32 -> 7x7x64, then flatten to a vector.
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
        )
        self.encoder_flat_dim = 64 * 7 * 7

        # Two separate linear heads: one predicts the mean (mu), one predicts the
        # log-variance (logvar) of the latent distribution q(z|x).
        # We predict log-variance instead of variance directly because it can be
        # any real number (variance must be positive), which is easier for a
        # network to output.
        self.fc_mu = nn.Linear(self.encoder_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flat_dim, latent_dim)

        # --- Decoder ---
        # Mirror of the encoder: latent vector -> 7x7x64 -> 14x14x32 -> 28x28x1
        self.decoder_fc = nn.Linear(latent_dim, self.encoder_flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid(),  # squashes output to [0, 1] to match normalized pixel values
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        The "reparameterization trick": we want to sample z ~ N(mu, sigma^2),
        but sampling directly is a non-differentiable operation and would block
        gradients from flowing back into the encoder during training.

        Instead we sample noise eps ~ N(0, 1) (which doesn't depend on the
        network's parameters) and compute z = mu + eps * std. Now z is a
        differentiable function of mu and logvar, so backprop works normally.
        """
        std = torch.exp(0.5 * logvar)  # logvar -> std: std = sqrt(exp(logvar))
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 7, 7)
        recon = self.decoder_conv(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    """
    VAE loss = reconstruction loss + KL divergence.

    Reconstruction loss: binary cross-entropy (BCE) between the reconstructed
    image and the original. We use BCE (rather than MSE) because pixel values
    are normalized to [0, 1] and the decoder ends in a sigmoid, so each pixel
    can be treated like a probability (this is the standard choice for MNIST
    VAEs). Summed over all pixels, then averaged over the batch.

    KL divergence: pulls the learned distribution q(z|x) = N(mu, sigma^2)
    towards the standard normal prior p(z) = N(0, 1). This is what keeps the
    latent space well-organized and makes random sampling produce sensible
    images later. Closed-form for two Gaussians:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Returns (total_loss, recon_loss, kl_loss) — all three so training can log
    each term separately.
    """
    batch_size = x.size(0)

    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss
