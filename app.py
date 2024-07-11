from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image
from tempfile import NamedTemporaryFile

from pokemon_stable_diffusion.latent_diffusion import LatentDiffusion
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.models.diffusion.ddim import DDIMSampler

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str

# Load model (do this at startup)
def load_model():
    # Load configuration
    model_config = OmegaConf.load('conf/ddpm_config.yaml')
    model_params = model_config.model.params
    
    # Instantiate the model
    model = LatentDiffusion(**model_params)

    # Load and set up the first stage model (VAE)
    first_stage_config = model_params.first_stage_config.params
    first_stage_model = AutoencoderKL(**first_stage_config)
    model.first_stage_model = first_stage_model

    # Set up the conditioning stage model (CLIP)
    model.cond_stage_model = FrozenCLIPEmbedder()
    
    # Load the pretrained weights
    checkpoint = torch.load("sd-v1-4-full-ema.ckpt", map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model

model = load_model()

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        device = next(model.parameters()).device
        
        # Generate image
        c = model.get_learned_conditioning([request.prompt])
        sampler = DDIMSampler(model)
        samples, _ = sampler.sample(S=50, conditioning=c, batch_size=1, shape=[4, 64, 64])
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Save image to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            save_image(x_samples[0], temp_file.name)
            temp_file_path = temp_file.name

        # Return the image file
        return FileResponse(temp_file_path, media_type="image/png", filename="generated_image.png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Latent Diffusion API 🔥🔥🔥. Use /generate to create images🦉🦉🦉."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
