import logging

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Importing the used dataset subclass is needed to reconstruct the dataset
from src.data.make_dataset import PokemonDataset  # noqa

# Import UNet and the diffusion sampling process
from src.models.simple_ddpm import DenoiseDiffusion  # noqa
from src.models.simple_unet import DiffusionUNet  # noqa
import wandb

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="training_config")
def main(cfg):
    # create run
    wandb.init(
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    # create data loader
    data_path = cfg.paths.training_data
    train_dataset = torch.load(data_path)  # TODO: add training labels to dataset
    logger.info(f"loaded training data from: {data_path}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.training_params.num_workers,
    )

    # instantiate lightning model
    class MyLightningModel(DiffusionUNet, LightningModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            criterion_cls = getattr(nn, cfg.training_params.criterion)
            self.criterion = criterion_cls(**cfg.training_params.criterion_params)

        def configure_optimizers(self):
            optimizer_cls = getattr(optim, cfg.training_params.optimizer)
            optimizer_params = cfg.training_params.optimizer_params
            optimizer = optimizer_cls(self.parameters(), **optimizer_params)
            return optimizer

        def training_step(self, batch, batch_id):
            data, target = batch
            preds = self(data)
            loss = self.criterion(preds, target)
            return loss

        def forward(self, *args):
            return DiffusionUNet.forward(self, *args)

    model = MyLightningModel(**cfg.model_params)

    # train model
    wandb_logger = WandbLogger(log_model="all")
    trainer = Trainer(
        max_epochs=cfg.training_params.n_epochs,
        log_every_n_steps=cfg.training_params.logging_freq,
        logger=wandb_logger,
        profiler="simple",
        accelerator=cfg.training_params.accelerator,
        devices=cfg.training_params.devices,
        strategy=cfg.training_params.strategy,
        num_nodes=cfg.training_params.num_nodes,
    )
    trainer.fit(model, train_dataloader)

    # save model locally
    model_path = f"{cfg.paths.model_folder}/{cfg.paths.model_name}.pt"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
