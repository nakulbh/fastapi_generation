from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
import os
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Text to Image API")

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Initialize Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    image: str  # Base64 encoded image
    seed: int    # Seed used for generation

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    try:
        # Set seed if provided
        if request.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator = generator.manual_seed(request.seed)
        else:
            generator = None
            
        # Generate image
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            generator=generator
        ).images[0]

        # Convert PIL Image to base64 string directly
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Get the seed that was used
        used_seed = generator.initial_seed() if generator else torch.seed()

        return GenerationResponse(
            image=img_str,
            seed=used_seed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during image generation: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
