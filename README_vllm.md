# SmolVLM2 (vLLM) Inference Setup

Use a separate environment for the vLLM server to avoid dependency clashes with RL-VLM-F training.

## 1) Create a dedicated env
```bash
conda create -n smolvlm-vllm python=3.10 -y
conda activate smolvlm-vllm
```

## 2) Install vLLM (choose the CUDA build that matches your driver)
- CUDA 12.2: `pip install "vllm>=0.6.4" vllm-cu122`
- CUDA 12.1: `pip install "vllm>=0.6.4" vllm-cu121`
- CUDA 11.8: `pip install "vllm>=0.6.4" vllm-cu118`
- CPU fallback (slow): `pip install "vllm>=0.6.4"`

Optional cache locations:
```bash
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
```

## 3) Launch OpenAI-compatible server (SmolVLM2-500M)
```bash
vllm serve HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --port 8000 \
  --api-key test-key
```
Keep this process running; first run will download model weights.

## 4) Call from RL-VLM-F (or any client) via OpenAI SDK
In your training env (not the vLLM env):
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="test-key")

resp = client.chat.completions.create(
    model="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///abs/path/img1.jpg"}},
                {"type": "image_url", "image_url": {"url": "file:///abs/path/img2.jpg"}},
                {"type": "text", "text": "Which image better matches the goal?"}
            ],
        }
    ],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```
Use `http://server-ip:8000` if vLLM runs on another machine.

## 5) Notes
- GPU with bfloat16 support recommended (A100/RTX 30+/L40 etc.).
- If you need GPT-4V or Gemini in the same training run, keep their API calls unchanged; this setup is only for SmolVLM2 via vLLM.
- RL-VLM-F integration: set `SMOLVLM_VLLM_URL` (default `http://localhost:8000/v1`) and `SMOLVLM_VLLM_API_KEY` (default `test-key`), then run with `vlm=smolvlm_vllm` and `image_reward=1, segment=1` to query the local SmolVLM2 service for preference labels.
