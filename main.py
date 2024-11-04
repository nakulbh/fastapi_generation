import logging
import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import shutil
import psutil
from pathlib import Path
import subprocess
import time
from fastapi.responses import FileResponse
import uuid

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add numpy import and check
try:
    import numpy as np
    logger.info(f"Numpy version: {np.__version__}")
    logger.info(f"Numpy configuration: {np.show_config()}")
except ImportError as e:
    logger.error(f"Numpy import error: {e}")
    raise ImportError("Numpy is required but not installed. Please install it with 'pip install numpy'")

app = FastAPI(title="Text to Image API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add these functions at the top of your file
def get_free_space_gb(directory):
    """Return free space in GB."""
    stats = shutil.disk_usage(directory)
    return stats.free / (2**30)  # Convert bytes to GB

def cleanup_cache():
    """Clean up the HuggingFace cache directory."""
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info("Cleaned up HuggingFace cache directory")

def get_disk_usage():
    """Get disk usage information."""
    try:
        df = subprocess.check_output(['df', '-h']).decode('utf-8')
        logger.info(f"Disk usage:\n{df}")
    except Exception as e:
        logger.error(f"Failed to get disk usage: {e}")

def aggressive_cleanup():
    """Aggressively clean up disk space."""
    directories_to_clean = [
        Path.home() / ".cache" / "huggingface",
        Path("/tmp/huggingface_cache"),
        Path("/tmp/huggingface_home"),
        Path("/tmp"),
        Path.home() / ".cache"
    ]
    
    for directory in directories_to_clean:
        if directory.exists():
            logger.info(f"Cleaning up directory: {directory}")
            try:
                if directory == Path("/tmp"):
                    # Only remove files older than 1 hour in /tmp
                    for item in directory.glob("*"):
                        if item.stat().st_mtime < time.time() - 3600:
                            try:
                                if item.is_file():
                                    item.unlink()
                                elif item.is_dir():
                                    shutil.rmtree(item)
                            except Exception as e:
                                logger.error(f"Failed to remove {item}: {e}")
                else:
                    shutil.rmtree(directory, ignore_errors=True)
                logger.info(f"Cleaned up {directory}")
            except Exception as e:
                logger.error(f"Failed to clean {directory}: {e}")

# Before initializing the pipeline, perform cleanup and checks
try:
    # Log initial disk space
    get_disk_usage()
    logger.info("Starting aggressive cleanup")
    aggressive_cleanup()
    get_disk_usage()  # Log disk space after cleanup
    
    # Set environment variables for cache location
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
    os.environ["HF_HOME"] = "/tmp/huggingface_home"
    
    # Create cache directories
    cache_dir = Path("/tmp/huggingface_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check available space
    free_space = get_free_space_gb("/tmp")
    logger.info(f"Available space in /tmp: {free_space:.2f} GB")
    
    if free_space < 5:  # Need at least 5GB
        raise RuntimeError(f"Insufficient disk space. Only {free_space:.2f} GB available")
    
    # Initialize pipeline with float32 for CPU
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 instead of float16
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=str(cache_dir),
        local_files_only=False
    )

    # Move to CPU and enable optimizations
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # Optional: Set lower precision for inputs while keeping model in float32
    pipe.unet.to(torch.float32)
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(torch.float32)

    logger.info("Pipeline initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    get_disk_usage()  # Log final disk space state
    raise HTTPException(
        status_code=500,
        detail=f"Failed to initialize pipeline due to disk space issues. Please contact administrator."
    )

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None
@app.post("/generate")
async def generate_image(request: GenerationRequest):
    logger.info("Received image generation request.")
    temp_file = None
    
    try:
        # Generate image with Stable Diffusion pipeline
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
        )
        
        image = result.images[0]
        logger.info("Image generated successfully")

        # Create temp file with unique name
        temp_file = f"/tmp/generated_image_{uuid.uuid4()}.png"
        
        # Save image and verify it exists
        image.save(temp_file, format="PNG")
        if not os.path.exists(temp_file):
            raise Exception("Failed to save image file")
        
        file_size = os.path.getsize(temp_file)
        logger.info(f"Image saved to {temp_file} (size: {file_size} bytes)")

        if file_size == 0:
            raise Exception("Generated image file is empty")

        # Return the file
        return FileResponse(
            path=temp_file,
            media_type="image/png",
            filename="generated_image.png",
            background=lambda: cleanup_temp_file(temp_file)
        )

    except Exception as e:
        logger.error(f"Error during image generation or sending: {str(e)}", exc_info=True)
        # Clean up temp file if it exists
        if temp_file and os.path.exists(temp_file):
            cleanup_temp_file(temp_file)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate or send image: {str(e)}"
        )

def cleanup_temp_file(file_path: str):
    """Safely cleanup temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Successfully deleted temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to delete temporary file {file_path}: {str(e)}")

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {"status": "healthy"}
