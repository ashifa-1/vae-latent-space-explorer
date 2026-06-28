import sys
import os

sys.path.append(os.path.abspath("."))

import torch
from app.models.vae import VAE

model = VAE(latent_dim=8)

dummy_input = torch.randn(4, 1, 28, 28)

reconstructed, mu, logvar = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Reconstruction shape:", reconstructed.shape)
print("Mu shape:", mu.shape)
print("LogVar shape:", logvar.shape)