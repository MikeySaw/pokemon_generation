import torch
from ts.torch_handler.base_handler import BaseHandler
from omegaconf import OmegaConf
from torchvision.utils import save_image
from io import BytesIO
import base64

from pokemon_stable_diffusion.latent_diffusion import LatentDiffusion
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.models.diffusion.ddim import DDIMSampler

class LatentDiffusionHandler(BaseHandler):
    def __init__(self):
        super(LatentDiffusionHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load configuration
        model_config = OmegaConf.load(f'{model_dir}/ddpm_config.yaml')
        model_params = model_config.model.params
        
        # Instantiate the model
        self.model = LatentDiffusion(**model_params)

        # Load and set up the first stage model (VAE)
        first_stage_config = model_params.first_stage_config.params
        first_stage_model = AutoencoderKL(**first_stage_config)
        self.model.first_stage_model = first_stage_model

        # Set up the conditioning stage model (CLIP)
        self.model.cond_stage_model = FrozenCLIPEmbedder()
        
        # Load the pretrained weights
        checkpoint = torch.load(f'{model_dir}/sd-v1-4-full-ema.ckpt', map_location='cpu')
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True

    def preprocess(self, data):
        prompt = data[0].get("data")
        if prompt is None:
            prompt = data[0].get("body")
        return prompt

    def inference(self, prompt):
        c = self.model.get_learned_conditioning([prompt])
        sampler = DDIMSampler(self.model)
        samples, _ = sampler.sample(S=50, conditioning=c, batch_size=1, shape=[4, 64, 64])
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples[0]

    def postprocess(self, inference_output):
        buffer = BytesIO()
        save_image(inference_output, buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return [{"generated_image": image_base64}]

    def handle(self, data, context):
        prompt = self.preprocess(data)
        generated_image = self.inference(prompt)
        return self.postprocess(generated_image)
    