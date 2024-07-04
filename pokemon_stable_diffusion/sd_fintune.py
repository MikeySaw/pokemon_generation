"""
In this .py file we implmented the training process for finetuning a stable diffusion model for generating pokemon images.
Some of the models we imported here are copied from the original latent diffusion model's repo:
https://github.com/CompVis/latent-diffusion
"""
# To Do: Need to change the yaml file loaded in this file to the correct version/correct format
# To Do: Need to implement the backbone models into correct folders, right now we will assume they are inside the model folder
# To Do: the model would be instantiated by the `instantiate_from_config` function, I think we could add more config files there
# just in case "equal contributions" are needed in our project
# To Do: figure out whether we only need to implement the `ddp` into inference or also training process
# To Do: integrate the pytorch profiler into the training process
# Import monitor libraries
import json
import random
import os
import wandb 
from tqdm import tqdm
from functools import partial
import argparse
import time
from ddpm_model import ddpm

# Import training libraries
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision import transforms
from omegaconf import OmegaConf

# Import Model Libraries
# from models.ldm.data.base import Txt2ImgIterableBaseDataset
from models.ldm.utils import instantiate_from_config

# Import the formatting libraries
from typing import List, Tuple, Optional
from jaxtyping import Array


def seed_everything(config:OmegaConf):
    """
    Seed everything for reproducibility.
    Args:
        config: The configuration file we want to use. -> OmegaConf
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

# define a placeholder data loading function
# to do: integrate this function with the `DatasetFromConfig` class
def get_dataset(config):
    pass

# Define a wrapper function to get the kwargs defined by argparse
# To Do: Need to add more arguments to the parser, right now it is just a placeholder
def get_parser(**kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    pass


# Define a helper function to change the shape into a UNet compatible shape
def modify_weights(w, scale = 1e-6, n=2):
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale*torch.randn_like(w)
    new_w = w.clone()
    for i in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w


# Define a helper function for reproducibility for each workers 
# The original ldm repo used ImageNet to train the model, we do not have to follow it.
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class CustomImageTextDataset(Dataset):
    """
    A very simplified version of the dataset class codes
    """
    def __init__(self, 
                 json_file: str, 
                 img_dir:str, 
                 transform: Optional[transforms.Compose]=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform or self._default_transform()

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'text': item['text']
        }

def get_data_loaders(config):
    train_dataset = CustomImageTextDataset(
        json_file=config['train_json'],
        img_dir=config['img_dir'],
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    val_dataset = CustomImageTextDataset(
        json_file=config['val_json'],
        img_dir=config['img_dir'],
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False, 
        worker_init_fn=None
    )

    return train_loader, val_loader


def setup_training(config:OmegaConf, output_dir:str):
    """
    set up the config file for the training process and the output directory.
    """
    output_dir = os.mkdir(output_dir, exist_ok=True)
    # save the config file if needed
    # OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))
    return output_dir

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim, epoch:int, path:str):
    """
    a function to save the trained checkpoint of the model.
    args:
        epoch: the number of the epoch when we save the model pretrained states.
        optimizer: the optimizer we used to train the model.
        model: the model we trained for generating the images.
        path: the path to save the state_dict of the model and the optimizer.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    },path)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim, path:str):
    """
    a function to load the checkpoints of the model pretrained weights and optimizer state_dict.
    args:
        model: the model we trained for generating the images.
        optimizer: the optimizer we used to train the model.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]

def log_images(images:torch.Tensor, step:int, output_dir:str=None, use_wandb:bool=True):
    """
    log the generated images, set up a flag to define whether we will log the generated images to wandb.
    args:
        images: the generated images.
        step: the current training epoch number.
        output_dir: the path to save the generated images.
        use_wandb: a flag to define whether we will log the generated images to wandb.
    """
    for k, v in images.items():
        if output_dir:
            image_path = os.path.join(output_dir, f"{k}_{step}.png")
            torchvision.utils.save_image(v, image_path)
        if use_wandb:
            wandb.log({k:[wandb.Image(image) for image in v]})

def get_cuda_status(start_time):
    """
    define a function to record the cuda status and the time it took to run one epoch.
    args:
        start_time: the time when the training process started.
    """
    torch.cuda.synchronize()
    cuda_peak_memory = torch.cuda.max_memory_allocated() / 2 ** 20
    elapsed_time = time.time() - start_time 
    
    # use this flag to check whether there is a wandb process running already
    if wandb.run is not None:
        wandb.log({'cuda_max_memory': cuda_peak_memory, 'elapsed_time': elapsed_time})
        
    return cuda_peak_memory, elapsed_time


# define the main function to train and test the model
def main():
    # Load configuration
    config = OmegaConf.load('config/default_config.yaml')
    
    # Initialize wandb, log all the hyperparameters loaded from the OmegeConf
    wandb.init(project=config.project_name, config=OmegaConf.to_container(config, resolve=True))
    # I felt like the following line will also work 
    # wandb.init(project=config.project_name, config=dict(config))    

    # Set device, this may need to be changed if the `ddp` is integrate into the training process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    # to do:need to load multiple datasets, right now there is only train data here.
    dataset = get_dataset(config.data)
    dataloader = DataLoader(dataset, 
                            batch_size=config.batch_size, 
                            shuffle=True, 
                            num_workers=config.num_workers, 
                            pin_memory=config.pin_memory)

    # Initialize model
    model = instantiate_from_config(config.model)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.min_lr)
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() # reset the peak memory stats to calculate the peak memory usage
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for batch in pbar:
                # Move batch to device
                batch = batch.to(device)

                # Forward pass
                loss = model(batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to stabilize training
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()

                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

                # Log to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # Step the scheduler
        scheduler.step()

        # Log epoch metrics
        wandb.log({
            "epoch_loss": epoch_loss / len(dataloader),
            "epoch": epoch
        })

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            wandb.save(checkpoint_path)

    wandb.finish()

if __name__ == "__main__":
    main()
