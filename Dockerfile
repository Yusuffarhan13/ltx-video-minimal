FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system dependencies
# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Colombo apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata && \
    rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /workspace/app

# Install Python dependencies
RUN pip install --no-cache-dir \
    diffusers[torch] \
    transformers \
    accelerate \
    fastapi \
    uvicorn[standard] \
    imageio[ffmpeg] \
    pydantic

# Pre-download the LTX Video model (this is the key optimization!)
# This downloads ~10-20GB but makes subsequent launches instant
RUN python -c "from diffusers import LTXPipeline; import torch; \
    print('Downloading LTX Video model...'); \
    model = LTXPipeline.from_pretrained('Lightricks/LTX-Video', torch_dtype=torch.bfloat16); \
    print('Model downloaded and cached!')"

# Copy server script
COPY ltx_video_server.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "ltx_video_server.py"]
