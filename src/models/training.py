import os
import logging
import time

import hydra

logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="training_config")
def main(cfg):
    logger.info(f"working directory: {os.getcwd()}")
    logger.info(f'working directory contents: {os.listdir(".")}')
    logger.info(f'contents of data:", {os.listdir("data")}')
    logger.info(f"loaded training data from: {cfg.paths.training_data}")
    logger.info(f"training model with {cfg.model_params.size} layers with batch size {cfg.training_params.batch_size}")
    for i in range(cfg.training_params.n_epochs):
        logger.info(f"   epoch {i}")
        time.sleep(0.5)
    logger.info(f"training of {cfg.paths.model_name} complete")
    logger.info(f"saving model in {cfg.paths.model_folder} complete")

if __name__ == "__main__":
    main()
