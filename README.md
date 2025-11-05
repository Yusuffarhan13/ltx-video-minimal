# LTX Video Server - Minimal

Minimal FastAPI server for LTX Video generation on Vast.ai GPU instances.

## What's Included
- `ltx_video_server.py` - FastAPI server for video generation
- `requirements.txt` - Python dependencies

## Setup on Vast.ai

```bash
git clone https://github.com/Yusuffarhan13/ltx-video-minimal.git /workspace/app
cd /workspace/app
pip install -r requirements.txt
python ltx_video_server.py
```

## API Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /generate` - Start video generation
- `GET /status/{job_id}` - Check job status
- `GET /download/{job_id}` - Download completed video

## Why Minimal?

This repo contains ONLY the files needed for LTX video generation.
No extra application code, no unnecessary dependencies.
Faster setup times on GPU instances.
