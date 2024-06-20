"""The training script for the DDPM diffusion model."""
# Construct the absolute path to the project's root directory
import os
import sys

import albumentations as A
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
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
import models.simple_ddpm as diffusion  # noqa
from models.simple_ddpm import DenoiseDiffusion as ddf
from models.simple_unet import DiffusionUNet  # noqa

# Load the config file by using the OmegaConf library
config = OmegaConf.load('../../conf/config.yaml')

eps_pred_model = DiffusionUNet(image_channels=3, n_channels=64)
print(eps_pred_model)

# Create all the things needed for training
train_data_loader = torch.utils.data.DataLoader(config.dataset,
                                                config.batch_size,
                                                shuffle=True,
                                                pin_memory=True)
optimizer = torch.optim.Adam(eps_pred_model.parameters(),
                             lr=config.training.learning_rate)

# Initialize the wandb
wandb.init(project='diffusion', entity='yecanlee', config=config)


def sample(config: OmegaConf):
    """Sample images from the model, this should be a slower version of generating images the trick would be related to
    the inference mode and torch.autocast."""
    with torch.no_grad():
        # Another trick of profiling, we may use the .to(device) first then use "device=config.device" later to show faster speed
        x = torch.randn([
            config.training.n_samples, config.model.image_channels,
            config.dataset.image_size, config.dataset.image_size
        ],
                        device=config.device)  # noqa
        for t_ in tqdm(range(config.diffusion.n_steps), desc='Sampling'):
            t = config.diffusion.n_steps - t_ - 1
            # Reverse pass denoising sampling process
            x = diffusion.p_sample(
                x,
                x.new_full((config.training.n_samples, ), t, dtype=torch.long))

        # Log samples to wandb
        wandb.log({'generated_samples': [wandb.Image(sample) for sample in x]})
        torchvision.utils.save_image(x, 'sample.png', nrow=4, normalize=True)


# define the training function
def train(config: OmegaConf):
    """
    Train a model for denosing diffusion probabilistic models.
    The core idea of this model would be add noise to the image at each step in 1000 steps, and then reverse the process
    """

    # Iterate through the dataset
    for data in train_data_loader:
        # Move data to device
        data = data.to(config.device)

        # Make the gradients zero
        config.optimizer.zero_grad()
        # Calculate loss
        loss = ddf.loss(data)
        # Compute gradients
        loss.backward()
        config.optimizer.step()
        wandb.log({'loss': loss.item()})


# define the inference function
def run(config: OmegaConf):
    """
    Run one epoch of training and inference process for the DDPM model.
    """
    train_whole_loss = 0
    for _ in tqdm(range(config.training.epochs), desc='Epochs'):
        train(config)
        sample(config)


# Define the image augmentation class
class AlbumentationsTransform:
    """
    A wrapper class for albumentations augmentations.
    """
    def __init__(self, transforms):
        self.transforms = A.Compose(transforms)

    def __call__(self, image):
        augmented = self.transforms(image=image)
        image = augmented['image']
        return image


# Define Custom Processor
class CustomProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image_path, text, max_length=100):
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(text, dtype=torch.long)
        }

# Define the Custom Dataset
class DiffusionTensorDataset(Dataset):
    def __init__(self, dataframe, root_dir, processor):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.root_dir, row['preproc_file_name'])
        text = row['label_eos']
        return self.processor(image_path, text, max_length=100)




def main():
    # Load the config file by using the OmegaConf library
    config = OmegaConf.load('../../conf/config.yaml')

    # Initialize the wandb
    wandb.init(project='diffusion', entity='yecanlee', config=config)

    # Run the training and inference process
    run(config)

    # Save the model
    torch.save(eps_pred_model.state_dict(), 'eps_pred_model.pth')

#
if __name__ == '__main__':
    main()