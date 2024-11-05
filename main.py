from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text to Image API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to("cpu")

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

@app.post("/generate")
async def generate_image(request: ImageRequest):
    try:
        # Generate image
        result = pipe(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
        )
        
        # Save image to temporary file
        temp_file = f"/tmp/generated_image_{uuid.uuid4()}.png"
        result.images[0].save(temp_file, format="PNG")
        
        # Return the image file
        return FileResponse(
            path=temp_file,
            media_type="image/png",
            filename="generated_image.png",
            background=lambda: os.remove(temp_file)  # Clean up after sending
        )

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}