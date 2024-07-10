import pytest
import torch

from pokemon_stable_diffusion.ddpm_model import DDPM


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

def test_ddpm_forward_pass_with_various_inputs(ddpm_model):
    input_shapes = [
        (1, 3, 32, 32),
        (2, 3, 32, 32),
        (4, 3, 32, 32)
    ]
    
    for shape in input_shapes:
        dummy_input = torch.randn(shape)
        dummy_batch = {ddpm_model.first_stage_key: dummy_input}
        try:
            output = ddpm_model(dummy_batch)
            assert output is not None, f"The model output is None for input shape {shape}"
        except Exception as e:
            pytest.fail(f"Model forward pass failed with input shape {shape} and exception: {e}")
