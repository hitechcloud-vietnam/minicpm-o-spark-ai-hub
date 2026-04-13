# minicpm-o-spark-ai-hub

Wrapper image of [MiniCPM-o 2.6](https://github.com/OpenBMB/MiniCPM-o) streaming
multimodal web demo for the NVIDIA DGX Spark (aarch64 / CUDA 13).

- Backend: `model_server.py` (FastAPI + WebSocket) on internal port 32550
- Frontend: Vue/Vite static bundle, served by nginx on port **7860**
- Image: `ghcr.io/waxacabytes/minicpm-o-spark-ai-hub:latest`
