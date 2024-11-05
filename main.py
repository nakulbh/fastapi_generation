from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
import logging
import gc
import psutil
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text to Image API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure PyTorch for CPU only
torch.set_num_threads(4)  # Limit CPU threads for efficiency
os.environ["TORCH_CPU_ALLOCATOR"] = "native"  # Use native memory allocator

# Initialize the model
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",  # Use a model optimized for CPU
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = pipe.to("cpu")  # Ensure the model is on CPU

    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_sequential_cpu_offload()

    gc.collect()  # Collect garbage to free memory
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 25  # Increased steps for better quality
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512  # Adjusted size for better resolution
    height: Optional[int] = 512  # Adjusted size for better resolution
    seed: Optional[int] = None

@app.post("/generate")
async def generate_image(request: ImageRequest):
    try:
        logger.info(f"Starting image generation for prompt: {request.prompt}")

        # Set random seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)

        gc.collect()  # Force garbage collection before generation

        # Generate image with memory optimizations
        with torch.no_grad(), torch.inference_mode():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
            )

        # Save image to temporary file
        temp_file = f"/tmp/generated_image_{uuid.uuid4()}.png"
        result.images[0].save(temp_file, format="PNG")

        logger.info("Image generated successfully")

        return FileResponse(
            path=temp_file,
            media_type="image/png",
            filename="generated_image.png",
            background=lambda: cleanup_file(temp_file)
        )
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image: {str(e)}"
        )

def cleanup_file(file_path: str):
    """Clean up the temporary file after response is sent."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {e}")

@app.get("/memory")
async def memory_status():
    """Endpoint to check memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "memory_used_mb": memory_info.rss / 1024 / 1024,
        "memory_percent": process.memory_percent(),
        "system_memory": dict(psutil.virtual_memory()._asdict())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify model and server status."""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "torch_threads": torch.get_num_threads()
    }
