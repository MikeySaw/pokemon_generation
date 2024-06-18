# The following three lines of codes show how to import the necessary modules to run the tests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__name__), 'models')))

import pytest
import torch
import torchvision
from torchvision.transforms import ToTensor

from models.simple_unet import DiffusionUNet

# Import a dataset from torchvision, then do some preprocessing, and then test the model
dataset = torchvision.datasets.CIFAR10(root="./data", download=True, transform=ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test initialization of the model
def test_simple_unet():
    model = DiffusionUNet(image_channels=3, n_channels=64)
    assert model is not None, "This model does not exist in the original file"

# Test the output of the model
def test_model_output():
    model = DiffusionUNet(image_channels=3, n_channels=64)
    model.to(device)
    t = torch.randn((1, 1), device=device)
    print(t.device)

    for x, _ in dataloader:
        with torch.inference_mode():
            x = x.to(device)
            out = model(x, t)
            assert out.shape == x.shape, "The output shape is not the same as the input shape"