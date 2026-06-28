import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    reconstruction_loss = F.binary_cross_entropy(
        recon_x,
        x,
        reduction="sum"
    )

    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + beta * kl_divergence

    return total_loss, reconstruction_loss, kl_divergence