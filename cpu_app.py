from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
import tempfile

# Add model monitoring
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)

class GenerationRequest(BaseModel):
    prompt: str

# Load model (do this at startup)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/sd-pokemon-diffusers", torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe, device

model, device = load_model()

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        # Generate image
        if device == "cuda":
            with torch.autocast("cuda"):
                image = model(request.prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        else:
            image = model(request.prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Return the image file
        return FileResponse(temp_file_path, media_type="image/png", filename="generated_image.png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Pokemon Stable Diffusion API. Use /generate to create images. Add /docs to the URL"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)