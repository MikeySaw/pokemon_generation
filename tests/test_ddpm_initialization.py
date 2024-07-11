import pytest
import torch
import torch.nn as nn

from pokemon_stable_diffusion.ddpm_model import DDPM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DummyModel(nn.Module):
    def __init__(self, image_size, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t, context=None):
        return self.relu(self.conv(x))


@pytest.mark.parametrize("image_size, channels, use_ema", [
    (32, 3, False),
    (64, 3, True),
    (128, 1, False),
    (256, 1, True),
])
def test_ddpm_initialization(image_size, channels, use_ema):
    unet_config = {
        'target': 'pokemon_stable_diffusion.ddpm_model.DiffusionWrapper',
        'params': {
            'diff_model_config': {
                'target': 'tests.test_ddpm_initialization.DummyModel',
                'params': {
                    'image_size': image_size,
                    'in_channels': channels,
                    'out_channels': channels,
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
        use_ema=use_ema,
        first_stage_key="image",
        image_size=image_size,
        channels=channels,
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
    ).to(device)
    assert model is not None
    dummy_input = torch.randn(1, channels, image_size, image_size).to(device)
    dummy_batch = {model.first_stage_key: dummy_input}
    output = model(dummy_batch)
    assert output is not None
