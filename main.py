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
    pipe.unet.dtype = torch.float32
    pipe.vae.dtype = torch.float32
    pipe.text_encoder.dtype = torch.float32

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

class GenerationResponse(BaseModel):
    image: str  # Base64 encoded image
    image_path: str  # Local path where image is saved

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    logger.info("Received image generation request.")
    logger.debug(f"Request parameters: {request}")

    # Check available disk space
    if get_free_space_gb("/tmp") < 1:  # Less than 1GB free
        cleanup_cache()
        if get_free_space_gb("/tmp") < 1:
            raise HTTPException(
                status_code=507,  # Insufficient Storage
                detail="Not enough disk space available for image generation"
            )

    # More aggressive parameter limits
    request.num_inference_steps = min(request.num_inference_steps or 20, 20)  # Max 20 steps
    request.width = min(request.width or 384, 384)  # Max 384x384
    request.height = min(request.height or 384, 384)
    request.guidance_scale = min(request.guidance_scale or 7.0, 7.0)  # Lower guidance scale
    
    logger.info(f"Adjusted parameters: steps={request.num_inference_steps}, size={request.width}x{request.height}")

    # Reduce default parameters for faster generation
    if request.num_inference_steps > 30:
        logger.warning("Reducing inference steps for performance")
        request.num_inference_steps = 30
    
    if request.width > 512 or request.height > 512:
        logger.warning("Reducing image dimensions for performance")
        request.width = min(request.width, 512)
        request.height = min(request.height, 512)
    
    try:
        # More detailed numpy verification
        try:
            test_array = np.zeros((1, 1))
            logger.info("Numpy verification successful")
        except Exception as e:
            logger.error(f"Numpy verification failed: {str(e)}")
            raise RuntimeError(f"Numpy verification failed: {str(e)}")
            
        # Set up generator with the provided seed, if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if request.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(request.seed)
            logger.info(f"Using specified seed: {request.seed}")
        else:
            generator = None
            logger.info("No seed provided; using default random seed.")

        # Generate image with Stable Diffusion pipeline
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            generator=generator
        ).images[0]

        # Save the generated image to a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "generated_image.png")
            image.save(temp_file_path, format="PNG")
            logger.info(f"Image saved to temporary file: {temp_file_path}")

            # Convert PIL Image to base64 string
            with open(temp_file_path, "rb") as f:
                img_str = base64.b64encode(f.read()).decode()

            # Determine and log the seed used for generation
            used_seed = generator.initial_seed() if generator else torch.seed()
            logger.info(f"Seed used for image generation: {used_seed}")

            return GenerationResponse(
                image=img_str,
                image_path=temp_file_path
            )

    except ValueError as e:
        logger.error(f"ValueError during image generation: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input parameters: {str(e)}")
    except RuntimeError as e:
        logger.error(f"RuntimeError during image generation: {e}")
        raise HTTPException(status_code=500, detail="Error in model computation; check model configuration and resources.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during image generation.")

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {"status": "healthy"}