import sys
import os

sys.path.append(os.path.abspath("."))

from app.utils.data_loader import get_dataloaders

train_loader, test_loader = get_dataloaders()

images, labels = next(iter(train_loader))

print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)