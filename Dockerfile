FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install static ffmpeg with libx264 support
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz \
    && tar -xf ffmpeg.tar.xz \
    && mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe \
    && rm -rf ffmpeg.tar.xz ffmpeg-*-amd64-static

# Fixes symlinks
RUN ln -sf /usr/local/bin/ffmpeg /opt/conda/bin/ffmpeg \
    && ln -sf /usr/local/bin/ffprobe /opt/conda/bin/ffprobe

RUN pip install --no-cache-dir --upgrade pip huggingface_hub

RUN mkdir -p /workspace/checkpoints

RUN huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir /workspace/checkpoints
RUN huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir /workspace/checkpoints

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir --retries 10 --timeout 100 numpy onnx onnxruntime opencv-python runpod

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "rp_handler.py"]