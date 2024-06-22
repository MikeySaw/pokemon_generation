import os
import sys
import albumentations as A
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import wandb
from PIL import Image
from omegaconf import OmegaConf
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

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

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


def setup(rank:int, world_size:int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    A helper function to destroy the process group
    """
    dist.destroy_process_group()


def sample(config:OmegaConf, ddpm:torch.nn.Module, rank:int):
    with torch.no_grad():
        x = torch.randn([
            config.training.n_samples, config.model.image_channels,
            config.dataset.image_size, config.dataset.image_size
        ], device=config.device)
        
        for t_ in tqdm(range(config.diffusion.n_steps), desc='Sampling'):
            t = config.diffusion.n_steps - t_ - 1
            x = ddpm.p_sample(x, x.new_full((config.training.n_samples,), t, dtype=torch.long))

    # Denormalize the images for visualization
    x = (x + 1) / 2

    if rank == 0: # log all the results only on the first GPU
        wandb.log({'generated_samples': [wandb.Image(sample) for sample in x]})
        torchvision.utils.save_image(x, f'sample_epoch_{wandb.run.step}.png', nrow=4, normalize=True)


# Config is not used but still added here, since I want to keep the function signature the same
def train(config, ddpm, optimizer, train_data_loader, rank):
    total_loss = 0
    for images, _ in train_data_loader:
        images = images.to(rank)
        optimizer.zero_grad()
        loss = ddpm.loss(images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if rank == 0:
            wandb.log({'batch_loss': loss.item()})
    return total_loss / len(train_data_loader)


def run(rank, world_size, config):
    setup(rank, world_size)
    
    # Set up augmentations
    transform = AlbumentationsTransform([
        A.RandomCrop(32, 32),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create the dataset and data loader
    dataset = AugmentedCIFAR10(root='./data', train=True, transform=transform, download=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=2)

    # Initialize the model and optimizer
    eps_pred_model = DiffusionUNet(image_channels=3, n_channels=config.model.n_channels).to(rank)
    eps_pred_model = DDP(eps_pred_model, device_ids=[rank])
    optimizer = torch.optim.Adam(eps_pred_model.parameters(), lr=config.training.learning_rate)

    # Create an instance of DenoiseDiffusion
    ddpm = ddf(eps_pred_model, rank, config.diffusion.n_steps)

    if rank == 0: # only log the results on the first GPU
        wandb.init(project='diffusion', config=OmegaConf.to_container(config, resolve=True))

    for epoch in tqdm(range(config.training.epochs), desc='Epochs', disable=rank!=0): # disable the tqdm bar for all GPUs except the first one
        sampler.set_epoch(epoch)
        train_loss = train(config, ddpm, optimizer, data_loader, rank)
        if rank == 0:
            wandb.log({'epoch': epoch, 'train_loss': train_loss})

        if (epoch + 1) % config.training.sample_interval == 0:
            sample(config, ddpm, rank)

    if rank == 0:
        # Save the model only to the first GPU
        torch.save(eps_pred_model.module.state_dict(), 'eps_pred_model.pth')

    cleanup()


def main():
    # Load the config file
    config = OmegaConf.load('../../conf/config.yaml')
    
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()