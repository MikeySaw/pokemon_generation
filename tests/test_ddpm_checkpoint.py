import os
import sys

import pytest
import torch

# Import the DDPM module
from pokemon_stable_diffusion.ddpm_model import DDPM

# Get the current directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add the project root to the path
sys.path.append(project_root)

# Ensure the correct paths are added
sys.path.append(os.path.join(project_root, 'pokemon_stable_diffusion'))

# Fixture to create a simple instance of the DDPM model
@pytest.fixture
def ddpm_model_config():
    model_config = {
        "unet_config": {
            "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "in_channels": 3,
                "out_channels": 3,
                "model_channels": 64,
                "channel_mult": [1, 2, 4],
                "num_res_blocks": 2,
                "attention_resolutions": [16],
                "dropout": 0.1,
                "image_size": 32,
                "num_heads": 4
            }
        },
        "timesteps": 1000,
        "beta_schedule": "linear",
        "loss_type": "l2",
        "monitor": "val/loss",
        "use_ema": False,
        "first_stage_key": "image",
        "image_size": 32,
        "channels": 3,
        "log_every_t": 100,
        "clip_denoised": True,
        "linear_start": 1e-4,
        "linear_end": 2e-2,
        "cosine_s": 8e-3,
        "original_elbo_weight": 0.,
        "v_posterior": 0.,
        "l_simple_weight": 1.,
        "conditioning_key": None,
        "parameterization": "eps",
        "scheduler_config": None,
        "use_positional_encodings": False,
        "learn_logvar": False,
        "logvar_init": 0.,
        "make_it_fit": False,
        "ucg_training": None
    }
    return model_config

@pytest.fixture
def ddpm_model(ddpm_model_config):
    return DDPM(**ddpm_model_config)

def test_ddpm_save_load_checkpoint(tmp_path, ddpm_model, ddpm_model_config):
    checkpoint_path = tmp_path / "ddpm_checkpoint.pth"
    torch.save(ddpm_model.state_dict(), checkpoint_path)
    
    loaded_model = DDPM(**ddpm_model_config)
    loaded_model.load_state_dict(torch.load(checkpoint_path))

    # Verify that the state dictionaries of the original and loaded models are the same
    for param_tensor in ddpm_model.state_dict():
        assert torch.equal(ddpm_model.state_dict()[param_tensor], loaded_model.state_dict()[param_tensor]), f"Mismatch found in {param_tensor}"

    # Run a forward pass to ensure the loaded model works as expected
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_batch = {ddpm_model.first_stage_key: dummy_input}
    output = loaded_model(dummy_batch)
    assert output is not None, "The loaded model's output is None"
