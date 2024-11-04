# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create cache directory and set permissions
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/logs && \
    useradd -m appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_CPU_ONLY=1
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"
ENV HF_HOME="/app/.cache/huggingface"

# Pre-download the model with explicit torch import
RUN python3 -c "import torch; from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float32)"

# Expose the port
EXPOSE 8000

# Start Gunicorn with proper logging
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "300", \
     "--worker-tmp-dir", "/dev/shm", \
     "--log-level", "debug", \
     "--error-logfile", "/app/logs/error.log", \
     "--access-logfile", "/app/logs/access.log", \
     "--capture-output", \
     "--enable-stdio-inheritance"]

