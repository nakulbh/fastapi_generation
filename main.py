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
import boto3
from typing import Optional
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

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
    num_inference_steps: Optional[int] = 20  # Changed to 20 as requested
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None

def upload_to_s3(file_path: str, object_name: str) -> str:
    """
    Upload a file to S3 bucket and return the URL
    """
    try:
        s3_client.upload_file(file_path, S3_BUCKET, object_name)
        url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{object_name}"
        logger.info(f"Successfully uploaded image to S3: {url}")
        return url
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image to S3: {str(e)}"
        )

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

        # Generate unique filename
        filename = f"generated_image_{uuid.uuid4()}.png"
        temp_file = f"/tmp/{filename}"
        
        # Save image to temporary file
        result.images[0].save(temp_file, format="PNG")

        # Upload to S3
        s3_url = upload_to_s3(temp_file, filename)

        logger.info("Image generated and uploaded successfully")

        # Cleanup temporary file
        cleanup_file(temp_file)

        # Return the S3 URL instead of the file
        return {"status": "success", "image_url": s3_url}

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
        "torch_threads": torch.get_num_threads(),
        "s3_configured": all([AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET])
    }