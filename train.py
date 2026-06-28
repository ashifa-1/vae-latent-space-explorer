import os
import torch
from torch import optim

from app.models.vae import VAE
from app.utils.losses import vae_loss
from app.utils.data_loader import get_dataloaders
from app.utils.logger import save_training_log


DEVICE = torch.device("cpu")

LATENT_DIM = 8
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

ANNEAL_EPOCHS = 3


def main():

    train_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = VAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    logs = []

    for epoch in range(EPOCHS):

        model.train()

        beta = min(
            1.0,
            (epoch + 1) / ANNEAL_EPOCHS
        )

        epoch_recon = 0.0
        epoch_kl = 0.0

        for images, _ in train_loader:

            images = images.to(DEVICE)

            optimizer.zero_grad()

            reconstructed, mu, logvar = model(images)

            loss, recon_loss, kl_loss = vae_loss(
                reconstructed,
                images,
                mu,
                logvar,
                beta
            )

            loss.backward()

            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Beta={beta:.2f} "
            f"Recon={avg_recon:.2f} "
            f"KL={avg_kl:.2f}"
        )

        logs.append({
            "epoch": epoch + 1,
            "reconstruction_loss": avg_recon,
            "kl_divergence": avg_kl
        })

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/vae.pt"
    )

    save_training_log(logs)

    print("\nTraining completed!")
    print("Model saved to models/vae.pt")
    print("Logs saved to results/training_log.csv")


if __name__ == "__main__":
    main()