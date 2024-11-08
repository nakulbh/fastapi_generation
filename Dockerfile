# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies including OpenMP
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set OpenMP environment variables
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

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
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "300", "--preload"]
