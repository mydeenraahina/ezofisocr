# Ezofisocr (RunPod Serverless)

[![Runpod](https://api.runpod.io/badge/mydeenraahina/ezofisocr)](https://console.runpod.io/hub/mydeenraahina/ezofisocr)

This repo is a RunPod **Serverless** worker that runs `handler.py` in a CUDA container.

## How to deploy on RunPod Hub (GitHub repo flow)

1. Push your code to GitHub (this repo).
2. In RunPod Hub, connect the repo.
3. Ensure these files exist (they do in this repo):
   - `Dockerfile`
   - `handler.py`
 
4. **Create a GitHub Release** (tag) — RunPod Hub requires a release to publish/update.
5. Deploy the released version in RunPod as **Serverless**.

## Inputs

Send either:

- `text_prompt`: string
- or `text`: string

## Optional environment variables

- `MODEL_NAME`: Hugging Face model id (default: `Qwen/Qwen3-VL-8B-Instruct`)
- `HF_TOKEN`: optional token (needed only if the model is gated/private)


