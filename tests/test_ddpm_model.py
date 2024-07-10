import pytest
import torch
from pokemon_stable_diffusion.ddpm_model import DDPM

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
                "image_size": 32,  # Ensure image_size is included
                "num_heads": 4  # Add this line to fix the issue
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
