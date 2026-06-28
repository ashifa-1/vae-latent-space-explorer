import torch

from app.utils.data_loader import get_dataloaders


def get_reconstruction(model):
    _, test_loader = get_dataloaders(batch_size=1)

    image, _ = next(iter(test_loader))

    with torch.no_grad():
        reconstructed, _, _ = model(image)

    return image.squeeze().numpy(), reconstructed.squeeze().numpy()