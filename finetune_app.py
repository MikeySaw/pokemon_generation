from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
import tempfile

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str

# Load model (do this at startup)
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/sd-pokemon-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

model = load_model()

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        # Generate image
        with torch.autocast("cuda"):
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
    