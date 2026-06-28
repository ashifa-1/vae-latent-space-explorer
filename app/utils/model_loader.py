import torch

from app.models.vae import VAE


def load_model(model_path="models/vae.pt", latent_dim=8):
    model = VAE(latent_dim=latent_dim)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    return model