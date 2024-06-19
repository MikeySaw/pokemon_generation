"""The training script for the DDPM diffusion model."""
# Construct the absolute path to the project's root directory
import os
import sys

import torch
import torch.utils.data
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


def sample():
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
