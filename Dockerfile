# =============================================================================
# MiniCPM-o 2.6 — NVIDIA DGX Spark (aarch64 / CUDA 13)
# Streaming multimodal demo: vision + voice Q&A over camera
# =============================================================================

# -------- Stage 1: build Vue frontend ---------------------------------------
FROM node:20-bookworm AS frontend-build

ARG MINICPM_REF=main
RUN apt-get update && apt-get install -y --no-install-recommends git openssl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/OpenBMB/MiniCPM-o.git /src && \
    cd /src && git checkout "${MINICPM_REF}"

WORKDIR /src/web_demos/minicpm-o_2.6/web_server

# vite.config.js reads cert.pem/key.pem at import time; generate throwaway
# ones so `pnpm run build` doesn't error. They are not used in production.
RUN openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \
        -days 365 -nodes -subj "/CN=localhost"

RUN npm install -g pnpm@9 && pnpm install && pnpm run build

# -------- Stage 2: runtime ---------------------------------------------------
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
        ffmpeg libsndfile1 libsndfile1-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        build-essential cmake \
        nginx \
    && rm -rf /var/lib/apt/lists/*

# Conda + Python 3.12
RUN curl -fsSL -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && rm -f /tmp/miniforge.sh

ENV PATH="/opt/conda/envs/minicpm/bin:/opt/conda/bin:${PATH}"
RUN conda create -y -n minicpm python=3.12 pip && conda clean -afy

# PyTorch cu130 aarch64 (proven on DGX Spark via acestep wrapper)
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    pip install --no-cache-dir \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cu130

# cuBLAS 13.0.2 in the base image crashes on GB10 (sm_121); override.
RUN pip install --no-cache-dir --index-url https://pypi.nvidia.com "nvidia-cublas>=13.1"
ENV LD_LIBRARY_PATH="/opt/conda/envs/minicpm/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"

# Clone upstream
ARG MINICPM_REF=main
RUN git clone https://github.com/OpenBMB/MiniCPM-o.git /app && \
    cd /app && git checkout "${MINICPM_REF}"

WORKDIR /app

# Python deps for the 2.6 web demo. Loosen torch pins (already installed),
# drop modelscope_studio (not used by this demo), leave librosa unpinned
# so aarch64 gets a compatible build.
RUN pip install --no-cache-dir \
      "Pillow>=10" \
      "transformers==4.44.2" \
      "sentencepiece==0.2.0" \
      "vector-quantize-pytorch==1.18.5" \
      "vocos==0.1.0" \
      "accelerate==1.2.1" \
      "timm==0.9.10" \
      "soundfile==0.12.1" \
      "librosa>=0.10" \
      "aiofiles==23.2.1" \
      "onnxruntime" \
      "fastapi" \
      "uvicorn[standard]" \
      "websockets" \
      "pydantic>=2.10" \
      "numpy<2"

# MiniCPM-o's remote modeling code has a top-level `import flash_attn` that
# transformers' dynamic_module_utils.check_imports enforces even when
# attn_implementation='sdpa' is explicitly selected. Ship an empty stub
# package so the import check passes; sdpa is the actual runtime path.
RUN python - <<'PY'
import os, site
sp = site.getsitepackages()[0]
pkg = os.path.join(sp, "flash_attn")
dist = os.path.join(sp, "flash_attn-2.7.4.post1.dist-info")
os.makedirs(pkg, exist_ok=True)
os.makedirs(dist, exist_ok=True)
with open(os.path.join(pkg, "__init__.py"), "w") as f:
    f.write('def __getattr__(name):\n    raise ImportError("flash_attn stub: real flash_attn not installed; attr=" + name)\n')
with open(os.path.join(dist, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: flash_attn\nVersion: 2.7.4.post1\n")
PY

# Frontend static bundle
COPY --from=frontend-build /src/web_demos/minicpm-o_2.6/web_server/dist /var/www/html

# Nginx config — listen on 7860, proxy /api/v1 and /ws to model_server on 32550
COPY nginx.conf /etc/nginx/nginx.conf

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# HuggingFace cache (mounted volume in compose)
ENV HF_HOME=/app/hf-cache
ENV MINICPM_MODEL=openbmb/MiniCPM-o-2_6

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
