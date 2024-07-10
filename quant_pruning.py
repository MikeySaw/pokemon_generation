# Import Model Libraries
from ldm.utils import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL #noqa
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from pokemon_stable_diffusion.latent_diffusion import LatentDiffusion # noqa
import time
import torch
from torch.nn.utils import prune

from omegaconf import OmegaConf


def measure_time(model, batch_size, device):
    dummy_captions = ["Old Italian photo"] * batch_size
    # Get text embeddings
    c = model.get_learned_conditioning(dummy_captions)

    tic = time.time()

    with torch.no_grad():
        dummy_images = torch.randn(batch_size, 3, 256, 256, device=device)
        encoder_posterior = model.encode_first_stage(dummy_images)
        z = model.get_first_stage_encoding(encoder_posterior).detach()

    model.train()
    _, _ = model.train_step({"image": dummy_images, "txt": dummy_captions}, c=c)
    toc = time.time()
    return(toc - tic)
    

def apply_pruning(module, amount=0.2):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.Linear):
            prune.l1_unstructured(child, name='weight', amount=amount)
        else: 
            apply_pruning(child)


def count_pruned_paramters(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.Linear):
            weight = child.weight
            pruned = torch.sum(weight == 0).item()
            total_params = weight.numel()
            print(f'{name}.weight: {pruned}/{total_params} elements are zeroed')
        else:
            count_pruned_paramters(child)
        

def main(path:str):
    # Load configuration
    config = OmegaConf.load(path)


    batch_size = 1

    device = torch.device(config.train.device) if config.train.device else \
        torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model_params = config.model.params
    
    # Instantiate the model
    model = LatentDiffusion(**model_params)

    # Load and set up the first stage model (VAE)
    first_stage_config = model_params.first_stage_config.params
    first_stage_model = AutoencoderKL(**first_stage_config)
    # first_stage_model.load_state_dict(torch.load(model_params.first_stage_config.ckpt_path))
    model.first_stage_model = first_stage_model

    # Set up the conditioning stage model (CLIP)
    model.cond_stage_model = FrozenCLIPEmbedder()
    
    # Load the pretrained weights into the model
    old_state = torch.load("sd-v1-4-full-ema.ckpt", map_location='cpu')
    old_state = old_state["state_dict"]

    model.first_stage_model = model.first_stage_model.to(device)
    model.cond_stage_model = model.cond_stage_model.to(device)
    model = model.to(device)

    time_before = measure_time(model=model, batch_size=batch_size, device=device)

    for name, module in model.named_modules():
        apply_pruning(module)

    # for name, module in model.named_modules():
    #     count_pruned_paramters(module)

    time_after = measure_time(model=model, batch_size=batch_size, device=device)
    print(f"Before pruning: {time_before}; After pruning: {time_after}")

    # the model is faster now 


if __name__ == "__main__":
    main("conf/ddpm_config.yaml")

