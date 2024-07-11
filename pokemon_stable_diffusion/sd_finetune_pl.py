import argparse
import os  # TODO: imported but unused
import sys  # TODO: imported but unused

from latent_diffusion_lightning import LatentDiffusion  # noqa
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sd_finetune import get_data_loaders
import torch

# Import Data Libraries
from torch.utils.data import DataLoader

from ldm.models.autoencoder import AutoencoderKL  # noqa
from ldm.modules.encoders.modules import FrozenCLIPEmbedder  # TODO: imported but unused

# Import Model Libraries
# from models.ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.utils import instantiate_from_config  # TODO: imported but unused
import wandb


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--train", type=bool, default=True, help="Run training")
    parser.add_argument("--test", type=bool, default=True, help="Run testing")
    parser.add_argument("--config_path", type=str, default="conf/ddpm_config.yaml", help="Path to config file")
    args = parser.parse_args()
    return args


def main(path: str):
    config = OmegaConf.load(path)
    pl.seed_everything(config.train.seed)

    wandb.init(
        **config.wandb,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    # Initialize WandB logger
    wandb_logger = WandbLogger(log_model="all")
    wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))  # correct, noqa

    # Prepare data
    train_set, _ = get_data_loaders(
        config, config.train.json_file_path, config.train.img_dir, config.train.batch_size, config.train.num_workers
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
    )
    # val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False,
    #                         num_workers=config.train.num_workers, pin_memory=config.train.pin_memory)

    # Initialize model
    model = LatentDiffusion(**config.model.params)

    # Load pretrained weights (if needed)
    if config.train.pretrained_path:
        old_state = torch.load(config.train.pretrained_path, map_location="cpu")
        model.load_state_dict(old_state["state_dict"], strict=False)

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config.train.num_epochs,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(every_n_epochs=config.train.save_every, save_top_k=-1),
            LearningRateMonitor(logging_interval="step"),
        ],
        accelerator="auto",  # Automatically choose GPU/CPU
        devices="auto",  # Use all available devices
        gradient_clip_val=config.train.max_grad_norm,
        precision=config.train.precision,  # Add this if you want to use mixed precision
        profiler="simple",
    )

    # Train the model
    trainer.fit(model, train_loader)

    wandb.finish()


if __name__ == "__main__":
    args = get_parser()
    main(args.config_path)
