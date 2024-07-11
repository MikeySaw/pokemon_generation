# add relative path to import the modules
import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parental directory, which should include both folders
project_root = os.path.dirname(current_dir)

# Add the ldm folder to the path
# DEBUG, remove the print if nothing happens
# print(f"Adding {project_root} to sys.path")
sys.path.append(project_root)

from ldm.utils import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL #noqa
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.models.diffusion.ddim import DDIMSampler
from pokemon_stable_diffusion.latent_diffusion import LatentDiffusion, save_image, make_grid # noqa
import torch
import argparse 
import numpy as np

from omegaconf import OmegaConf


def load_model(model_params, device, version):
    # Instantiate the model
    model_config = OmegaConf.load('conf/ddpm_config.yaml')
    model_params = model_config.model.params
    
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = LatentDiffusion(**model_params)

    # Load and set up the first stage model (VAE)
    first_stage_config = model_params.first_stage_config.params
    first_stage_model = AutoencoderKL(**first_stage_config)
    # first_stage_model.load_state_dict(torch.load(model_params.first_stage_config.ckpt_path))
    model.first_stage_model = first_stage_model

    # Set up the conditioning stage model (CLIP)
    model.cond_stage_model = FrozenCLIPEmbedder()
    
    if version == "base":
        state = torch.load("sd-v1-4-full-ema.ckpt", map_location='cpu')
        state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    else:
        pass

    # Move to the device!
    model.first_stage_model = model.first_stage_model.to(device)
    model.cond_stage_model = model.cond_stage_model.to(device)
    model = model.to(device)

    return model


def generate_sample(model, batch_size, c):
    model.eval()
    with torch.no_grad():
        sampler = DDIMSampler(model)
        samples, _ = sampler.sample(S=50, conditioning=c, batch_size=batch_size, shape=[4, 64, 64])
        print(f"Generated samples shape: {samples.shape}")
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        for i, x_sample in enumerate(x_samples):
            save_image(x_sample, f'sample_{i}.png')

    grid = make_grid(x_samples, nrow=int(np.sqrt(batch_size)))
    save_image(grid, 'samples_grid.png')


def main(path):
    parser = argparse.ArgumentParser(description="Prompt for the generation")
    parser.add_argument('prompt', type=str, help="Input a prompt here for the generation")
    parser.add_argument('--batch_size', type=int, help="Number of images to be generated")
    parser.add_argument('--model_version', type=str, help="Version has to be one of 'finetune' and 'base'")
    args = parser.parse_args()

    config = OmegaConf.load(path)
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captions = [args.prompt] * args.batch_size
    model_params = config.model.params

    model = load_model(model_params=model_params, device=device, version=args.model_version)

    c = model.get_learned_conditioning(captions)

    generate_sample(model=model, batch_size=args.batch_size, c=c)


if __name__ == "__main__":
    main("conf/ddpm_config.yaml")
