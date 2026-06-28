import sys
import os

sys.path.append(os.path.abspath("."))

import torch

from app.models.vae import VAE
from app.utils.losses import vae_loss


model = VAE(latent_dim=8)

dummy_input = torch.rand(4, 1, 28, 28)

reconstructed, mu, logvar = model(dummy_input)

loss, recon_loss, kl_loss = vae_loss(
    reconstructed,
    dummy_input,
    mu,
    logvar,
    beta=1.0
)

print("Total Loss:", loss.item())
print("Reconstruction Loss:", recon_loss.item())
print("KL Divergence:", kl_loss.item())