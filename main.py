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

# Initialize Stable Diffusion pipeline with more aggressive optimizations
try:
    model_id = "CompVis/stable-diffusion-v1-4"  # Smaller model
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Torch version: {torch.__version__}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to("cpu")
    
    # Enable all memory optimizations
    pipe.enable_attention_slicing(slice_size=1)  # Most aggressive slicing
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()  # Offload to CPU when possible
    
    logger.info("Stable Diffusion pipeline initialized with aggressive optimizations")
except Exception as e:
    logger.error(f"Failed to initialize Stable Diffusion pipeline: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error initializing Stable Diffusion pipeline: {str(e)}")

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