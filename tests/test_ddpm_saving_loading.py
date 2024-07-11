import os

import pytest
import torch

from pokemon_stable_diffusion.ddpm_model import DDPM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def ddpm_model():
    # Create a dummy unet_config that the model expects
    unet_config = {
        'target': 'pokemon_stable_diffusion.ddpm_model.DiffusionWrapper',  # Adjusted target
        'params': {
            'diff_model_config': {
                'target': 'ldm.modules.diffusionmodules.openaimodel.UNetModel',  # Use an actual class
                'params': {
                    'image_size': 32,
                    'in_channels': 3,
                    'out_channels': 3,
                    'model_channels': 64,
                    'num_res_blocks': 2,
                    'attention_resolutions': [16],
                    'dropout': 0.1,
                    'channel_mult': (1, 2, 4, 8),
                    'num_heads': 4,
                    'use_scale_shift_norm': True,
                }
            },
            'conditioning_key': None
        }
    }

    model = DDPM(
        unet_config=unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=False,
        first_stage_key="image",
        image_size=32,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.,
        v_posterior=0.,
        l_simple_weight=1.,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.,
        make_it_fit=False,
        ucg_training=None,
    )
    return model.to(device)

def test_model_saving_loading(ddpm_model, tmp_path):
    # Save the model
    save_path = tmp_path / "ddpm_model.pth"
    torch.save(ddpm_model.state_dict(), save_path)
    assert os.path.exists(save_path), "Model file not saved."

    # Load the model
    loaded_model = DDPM(
        unet_config={
            'target': 'pokemon_stable_diffusion.ddpm_model.DiffusionWrapper',  # Adjusted target
            'params': {
                'diff_model_config': {
                    'target': 'ldm.modules.diffusionmodules.openaimodel.UNetModel',  # Use an actual class
                    'params': {
                        'image_size': 32,
                        'in_channels': 3,
                        'out_channels': 3,
                        'model_channels': 64,
                        'num_res_blocks': 2,
                        'attention_resolutions': [16],
                        'dropout': 0.1,
                        'channel_mult': (1, 2, 4, 8),
                        'num_heads': 4,
                        'use_scale_shift_norm': True,
                    }
                },
                'conditioning_key': None
            }
        },
        timesteps=ddpm_model.num_timesteps,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=False,
        first_stage_key="image",
        image_size=32,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.,
        v_posterior=0.,
        l_simple_weight=1.,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.,
        make_it_fit=False,
        ucg_training=None
    )
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.to(device)

    # Check if the state dicts are the same
    for param_tensor in ddpm_model.state_dict():
        assert torch.equal(ddpm_model.state_dict()[param_tensor], loaded_model.state_dict()[param_tensor]), f"Mismatch in {param_tensor}"
