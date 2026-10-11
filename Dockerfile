FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install minimal OS dependencies for computer vision and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement definitions first for caching
COPY requirements.txt .

# Install CPU-optimized PyTorch and pip packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure upload, cache, and db directories have proper write permissions
RUN mkdir -p uploads static/xai static/css && \
    chmod -R 777 /app/uploads /app/static /app 2>/dev/null || true

# Expose port (HF Spaces defaults to 7860, Render/Railway pass dynamic PORT)
ENV PORT=7860
EXPOSE 7860

# Launch uvicorn web server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
