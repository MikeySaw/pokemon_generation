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
import os
import wandb 
from tqdm import tqdm
from functools import partial
import argparse
import time

# Import training libraries
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from omegaconf import OmegaConf

# Import Model Libraries
from models.ldm.data.base import Txt2ImgIterableBaseDataset
from models.ldm.utils import instantiate_from_config

# Import the formatting libraries
from typing import List, Tuple, Optional
from jaxtyping import Array


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


# define a class to load the train/test dataset from config files
class DatasetFromConfig(Dataset):
    """
    Define a class to load the corresponding dataset from the config file.
    Args:
        config: The configuration file. -> OmegaConf
    """
    def __init__(self, batch_size, train=None, validation=None, test=None, inference=None,
                 wrap=False, num_workers:int=16, shuffle_test_loader=False, shuffle_val_dataloader=False, 
                 num_val_workers=None, config:Optional[OmegaConf]=None):
        super(DatasetFromConfig, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dict_config = {}
        if self.num_val_workers is None:
            self.num_val_workers = num_workers
        else:
            self.num_val_workers = num_val_workers
        self.shuffle_test_loader = shuffle_test_loader
        self.shuffle_val_dataloader = shuffle_val_dataloader
        if train is not None:
            self.dict_config['train'] = train
            self.train_loader = self._train_loader
        if validation is not None:
            self.dict_config['validation'] = validation
            self.validation_loader = partial(self._validation_loader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dict_config['test'] = test
            self.test_loader = partial(self._test_loader, shuffle=shuffle_test_loader)
        if inference is not None:
            self.dict_config['inference'] = inference
            self.inference_loader = self._inference_loader()
        self.wrap = wrap
    
    def load_dataset(self):
        for config in self.dict_config.values():
            instantiate_from_config(config)
    
    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dict_config[k]))
            for k in self.dict_config)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_loader(self):
        iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)    
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, 
                        num_workers=self.num_workers, pin_memory=True, shuffule=False if iterable_dataset else True)
        
    def _validation_loader(self):
        return DataLoader(self.datasets['validation'], batch_size=self.batch_size,
                          num_workers=self.num_val_workers, pin_memory=True, shuffle=self.shuffle_val_dataloader)
    
    def _test_loader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=self.shuffle_test_loader)
    
    def _inference_loader(self):
        return DataLoader(self.datasets['inference'], batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=False)

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
