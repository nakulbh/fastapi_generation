from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
import os
import tempfile
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Custom exception for model loading errors"""
    pass

class ImageGenerationError(Exception):
    """Custom exception for image generation errors"""
    pass

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(default=None, max_length=1000)
    num_inference_steps: Optional[int] = Field(default=20, ge=1, le=150)
    guidance_scale: Optional[float] = Field(default=7.5, ge=1.0, le=20.0)
    width: Optional[int] = Field(default=512, ge=256, le=1024)
    height: Optional[int] = Field(default=512, ge=256, le=1024)
    seed: Optional[int] = Field(default=None, ge=0, lt=2**32)

class GenerationResponse(BaseModel):
    image: str
    seed: int
    generation_time: float

app = FastAPI(
    title="Text to Image API",
    description="Stable Diffusion text to image generation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def handle_temporary_file():
    """Context manager for handling temporary files"""
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        yield temp_file
    finally:
        if temp_file:
            try:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)
            except OSError as e:
                logger.error(f"Error cleaning up temporary file: {e}")

def initialize_pipeline():
    """Initialize the Stable Diffusion pipeline with error handling"""
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cpu"
        torch_dtype = torch.float32
        
        logger.info(f"Initializing Stable Diffusion pipeline on {device}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None
        )
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        return pipe
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise ModelLoadError(f"Failed to initialize model: {str(e)}")

# Initialize the pipeline
try:
    pipe = initialize_pipeline()
except ModelLoadError as e:
    logger.error(f"Critical error during pipeline initialization: {e}")
    raise

@app.post("/generate", 
         response_model=GenerationResponse,
         description="Generate an image from a text prompt")
async def generate_image(request: GenerationRequest):
    import time
    start_time = time.time()
    
    try:
        # Set up generator with seed
        device = "cpu"
        generator = torch.Generator(device=device)
        seed = request.seed if request.seed is not None else generator.seed()
        generator = generator.manual_seed(seed)
        
        logger.info(f"Starting image generation with prompt: {request.prompt[:50]}...")
        
        # Generate image with error handling
        try:
            with torch.inference_mode():
                output = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    width=request.width,
                    height=request.height,
                    generator=generator
                )
            image = output.images[0]
        except torch.cuda.OutOfMemoryError:
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory. Try reducing image dimensions or batch size."
            )
        except Exception as e:
            raise ImageGenerationError(f"Image generation failed: {str(e)}")

        # Save and encode image
        with handle_temporary_file() as temp_file:
            try:
                # Save image to temporary file
                image.save(temp_file.name, format="PNG")
                
                # Read and encode to base64
                with open(temp_file.name, "rb") as f:
                    img_str = base64.b64encode(f.read()).decode()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing generated image: {str(e)}"
                )

        generation_time = time.time() - start_time
        
        return GenerationResponse(
            image=img_str,
            seed=seed,
            generation_time=round(generation_time, 2)
        )

    except ImageGenerationError as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during image generation"
        )

@app.get("/health")
async def health_check():
    try:
        # Basic pipeline check
        if pipe is None:
            raise HTTPException(
                status_code=503,
                detail="Model pipeline not initialized"
            )
        
        # Memory check
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            gpu_info = {
                "allocated_memory_mb": round(memory_allocated, 2),
                "reserved_memory_mb": round(memory_reserved, 2)
            }
        else:
            gpu_info = None
        
        return {
            "status": "healthy",
            "backend": "CPU",
            "gpu_info": gpu_info,
            "model_loaded": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    logger.info("API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down...")
    # Clean up resources if needed
    torch.cuda.empty_cache()
