"""The training script for the DDPM diffusion model."""
# Construct the absolute path to the project's root directory
import os
import sys

import albumentations as A
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import wandb
from omegaconf import OmegaConf
from PIL import Image  # noqa
from tqdm import tqdm

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the project's root directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the project's root directory to the Python module search path
sys.path.append(project_root)

from typing import List  # noqa

# Import UNet and the diffusion sampling process
from models.simple_ddpm import DenoiseDiffusion as ddf
from models.simple_unet import DiffusionUNet  # noqa

# Add torch debug support
torch.autograd.set_detect_anomaly(True)

# Define the sampling function
def sample(config: OmegaConf, ddpm: torch.nn.Module):
    with torch.no_grad():
        x = torch.randn([
            config.training.n_samples, config.model.image_channels,
            config.dataset.image_size, config.dataset.image_size
        ], device=config.device)
        
        for t_ in tqdm(range(config.diffusion.n_steps), desc='Sampling'):
            t = config.diffusion.n_steps - t_ - 1
            x = ddpm.p_sample(x, x.new_full((config.training.n_samples,), t, dtype=torch.long))

    # Denormalize the images
    x = (x + 1) / 2

    # Log samples to wandb
    wandb.log({'generated_samples': [wandb.Image(sample) for sample in x]})
    torchvision.utils.save_image(x, f'sample_epoch_{wandb.run.step}.png', nrow=4, normalize=True)


# Define the training function
def train(config: OmegaConf, ddpm: torch.nn.Module, optimizer: torch.optim.Optimizer, train_data_loader: torch.utils.data.DataLoader):
    total_loss = 0
    for images, _ in train_data_loader:
        images = images.to(config.device)

        optimizer.zero_grad()
        loss = ddpm.loss(images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({'batch_loss': loss.item()})

    return total_loss / len(train_data_loader)


# define the training process wrapper
def run(config: OmegaConf, eps_pred_model: nn.Module, optimizer: torch.optim.Optimizer, train_data_loader: torch.utils.data.DataLoader):
    # Create an instance of DenoiseDiffusion
    ddpm = ddf(eps_pred_model, config.device, config.diffusion.n_steps)

    for epoch in tqdm(range(config.training.epochs), desc='Epochs'):
        train_loss = train(config, ddpm, optimizer, train_data_loader)
        wandb.log({'epoch': epoch, 'train_loss': train_loss})

        if (epoch + 1) % config.training.sample_interval == 0:
            sample(config, ddpm)


# Define the image augmentation class
class AlbumentationsTransform:
    def __init__(self, transforms: dict):
        """
        Args:
            transforms (Dict): a dictionary of albumentation transforms
        """
        self.transforms = A.Compose(transforms)

    def __call__(self, image: torch.Tensor):
        # first two lines were added to make sure no bug with a PIL image
        transform_to_tensor = transforms.ToTensor()
        image_tensor = transform_to_tensor(image)
        image = image_tensor.permute(1, 2, 0).numpy()  # Convert to numpy array
        augmented = self.transforms(image=image)
        image = augmented['image']
        return torch.from_numpy(image).permute(2, 0, 1)  # Convert back to tensor


# Define a Custom Dataset class for easy testing, later should be replaced by our real dataset
class AugmentedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.cifar10 = torchvision.datasets.CIFAR10(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    # Load the config file
    config = OmegaConf.load('../../conf/config.yaml')

    # Convert OmegaConf to a plain Python dictionary, this is needed for wandb logging
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Set up augmentations
    transform = AlbumentationsTransform([
        A.RandomCrop(32, 32),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create the dataset and data loader
    dataset = AugmentedCIFAR10(root='./data', train=True, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=2)

    # Initialize the model and optimizer
    eps_pred_model = DiffusionUNet(image_channels=3, n_channels=config.model.n_channels).to(config.device)

    # Load the pre-trained model if the path exists
    # if os.path.exists('eps_pred_model.pth'):
    #     eps_pred_model.load_state_dict(torch.load('eps_pred_model.pth'))
    optimizer = torch.optim.Adam(eps_pred_model.parameters(), lr=config.training.learning_rate)

    # Initialize wandb
    wandb.init(project='diffusion', config=config_dict)

    # Run the training and inference process
    run(config, eps_pred_model, optimizer, data_loader)

    # Save the model
    torch.save(eps_pred_model.state_dict(), 'eps_pred_model.pth')

if __name__ == '__main__':
    main()
