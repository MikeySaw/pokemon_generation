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
# Import monitor libraries
import wandb 
from tqdm import tqdm
from functools import partial
import argparse

# Import training libraries
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf

# Import Model Libraries
from models.ldm.data.base import Txt2ImgIterableBaseDataset
from models.ldm.utils import instantiate_from_config


def seed_everything(config):
    """
    Seed everything for reproducibility.
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

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


class WrappedDataset(Dataset):
    """
    A warpper class for the dataset to make it compatible with the DataLoader.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


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
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for batch in pbar:
                # Move batch to device
                batch = batch.to(device)

                # Forward pass
                loss = model(batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
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
