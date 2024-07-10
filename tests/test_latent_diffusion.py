# tests/test_diffusion.py
import sys
import os
import pytest
import torch

# Get the current directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add the project root to the path
sys.path.append(project_root)

# Ensure the correct paths are added
sys.path.append(os.path.join(project_root, 'pokemon_stable_diffusion'))

# Import the DDPM and LatentDiffusion modules
from pokemon_stable_diffusion.ddpm_model import DDPM
from pokemon_stable_diffusion.latent_diffusion import LatentDiffusion

# Fixture to create a simple instance of the DDPM model
@pytest.fixture
def ddpm_model():
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
    return DDPM(**model_config)

# Fixture to create a simple instance of the LatentDiffusion model
@pytest.fixture
def latent_diffusion_model():
    model_config = {
        "first_stage_config": {
            "target": "ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0
                },
                "lossconfig": {
                    "target": "torch.nn.Identity"
                }
            }
        },
        "cond_stage_config": {
            "target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder"
        },
        "num_timesteps_cond": 1,
        "cond_stage_key": "image",
        "cond_stage_trainable": False,
        "concat_mode": True,
        "cond_stage_forward": None,
        "conditioning_key": None,
        "scale_factor": 0.18215,
        "scale_by_std": True,
        "unet_trainable": True,
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
    return LatentDiffusion(**model_config)

# Tests for DDPM model
def test_ddpm_initialization(ddpm_model):
    assert ddpm_model is not None, "Failed to initialize the DDPM model"

def test_ddpm_forward_pass(ddpm_model):
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_batch = {ddpm_model.first_stage_key: dummy_input}
    try:
        output = ddpm_model(dummy_batch)
        assert output is not None, "The model output is None"
    except Exception as e:
        pytest.fail(f"Model forward pass failed with exception: {e}")

def test_ddpm_sample(ddpm_model):
    try:
        sample_output = ddpm_model.sample(batch_size=2)
        assert sample_output is not None, "Sampling did not produce any output"
    except Exception as e:
        pytest.fail(f"Sampling failed with exception: {e}")

def test_ddpm_loss_calculation(ddpm_model):
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_batch = {ddpm_model.first_stage_key: dummy_input}
    t = torch.randint(0, ddpm_model.num_timesteps, (dummy_input.shape[0],)).long()
    try:
        loss, _ = ddpm_model.p_losses(dummy_batch[ddpm_model.first_stage_key], t)
        assert loss is not None, "Loss calculation failed"
    except Exception as e:
        pytest.fail(f"Loss calculation failed with exception: {e}")

# Tests for LatentDiffusion model
def test_latent_diffusion_initialization(latent_diffusion_model):
    assert latent_diffusion_model is not None, "Failed to initialize the LatentDiffusion model"

def test_latent_diffusion_forward_pass(latent_diffusion_model):
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_batch = {latent_diffusion_model.first_stage_key: dummy_input}
    dummy_cond = torch.randn(1, 3, 256, 256)
    try:
        output = latent_diffusion_model(dummy_batch, dummy_cond)
        assert output is not None, "The model output is None"
    except Exception as e:
        pytest.fail(f"Model forward pass failed with exception: {e}")

def test_latent_diffusion_sample(latent_diffusion_model):
    dummy_cond = torch.randn(1, 3, 256, 256)
    try:
        sample_output = latent_diffusion_model.sample(cond=dummy_cond, batch_size=2)
        assert sample_output is not None, "Sampling did not produce any output"
    except Exception as e:
        pytest.fail(f"Sampling failed with exception: {e}")

def test_latent_diffusion_loss_calculation(latent_diffusion_model):
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_batch = {latent_diffusion_model.first_stage_key: dummy_input}
    dummy_cond = torch.randn(1, 3, 256, 256)
    t = torch.randint(0, latent_diffusion_model.num_timesteps, (dummy_input.shape[0],)).long()
    try:
        loss, _ = latent_diffusion_model.p_losses(dummy_batch[latent_diffusion_model.first_stage_key], dummy_cond, t)
        assert loss is not None, "Loss calculation failed"
    except Exception as e:
        pytest.fail(f"Loss calculation failed with exception: {e}")
