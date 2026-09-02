"""
GAN architecture: a Generator and a Discriminator, both simple MLPs.

This follows the "vanilla GAN" design from the PyTorch-GAN reference repo
(eriklindernoren/PyTorch-GAN, implementations/gan/gan.py) — no convolutions,
just fully-connected layers. It's the simplest possible GAN, which makes it
a good starting point for understanding adversarial training before moving
on to DCGAN/StyleGAN-style architectures.

- Generator: takes a random noise vector (the "latent" vector) and maps it
  to a fake image. Output uses Tanh so pixel values land in [-1, 1], which
  is why we normalize real images the same way in dataset.py.
- Discriminator: takes an image (real or fake) and outputs a single
  probability that the image is real, via Sigmoid.
"""

import math

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        img_size = int(math.prod(img_shape))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # BatchNorm stabilizes training by keeping activations from
                # exploding/vanishing as the generator gets deeper.
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_size),
            nn.Tanh(),  # squashes output to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        img_size = int(math.prod(img_shape))

        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # outputs a "real" probability between 0 and 1
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
