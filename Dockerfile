FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV / FastAI and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Runtime limits to reduce CPU/RAM pressure on small instances
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PREDICT_TIMEOUT_SECONDS=50 \
    MAX_IMAGE_DIM=1024 \
    MALLOC_ARENA_MAX=2 \
    ATEN_CPU_CAPABILITY=default \
    MKL_SERVICE_FORCE_INTEL=1

# Copy requirements
COPY requirements.txt .

# Install pinned CPU PyTorch stack for predictable behavior
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Install pinned fastai
RUN pip install --no-cache-dir fastai==2.8.7

# Install remaining requirements and gdown
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gdown

# Copy the application code
COPY . .

# Download model from Google Drive into the location expected by main.py
RUN gdown 1ppniUVWmgfNg_wnLFwx5YA-rk6mYQkMB -O /app/export.pkl

# Expose default app port
EXPOSE 7860

# Railway uses PORT at runtime; fallback to 7860 locally
CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 10'
