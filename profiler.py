import os
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parental directory, which should include both folders
project_root = os.path.dirname(current_dir)

# Add the ldm folder to the path
sys.path.append(project_root)

import torch
from ldm.models.diffusion.ddpm import DDPM


def main():
    # Load configuration
    config_path = os.path.join(project_root, 'conf', 'ddpm_config.yaml')
    config = torch.load(config_path)

    # Instantiate the model
    model_config = config["model"]["params"]["unet_config"]
    model = DDPM(**model_config)
    
    # Check if CUDA is available and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Dummy input
    dummy_input = torch.randn(1, 4, 32, 32).to(device)
    
    # Forward pass
    _ = model(dummy_input)


if __name__ == "__main__":
    main()
