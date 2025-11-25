import os
import tempfile
import argparse
from typing import Optional

import numpy as np
from PIL import Image
from openai import OpenAI


def save_temp_image(img: np.ndarray) -> str:
    """Save HWC uint8 image to a temp png and return file:// path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(img).save(tmp.name)
    return f"file://{tmp.name}"


def parse_pref_text(text: str) -> int:
    text = text.strip()
    if text == "0":
        return 0
    if text == "1":
        return 1
    return -1


def smolvlm_pair_preference(
    img1: np.ndarray,
    img2: np.ndarray,
    prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> int:
    """
    Query a local vLLM OpenAI-compatible server running SmolVLM2 for
    pairwise preference. Returns 0 if img1 better, 1 if img2 better,
    -1 on error/unknown.
    """
    base_url = base_url or os.getenv("SMOLVLM_VLLM_URL", "http://localhost:8000/v1")
    api_key = api_key or os.getenv("SMOLVLM_VLLM_API_KEY", "test-key")

    # Save images to temp files; ensure cleanup.
    paths: list[str] = []
    try:
        paths = [save_temp_image(img1), save_temp_image(img2)]

        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": paths[0]}},
                        {"type": "image_url", "image_url": {"url": paths[1]}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=16,
        )
        text = resp.choices[0].message.content.strip()
        return parse_pref_text(text)
    except Exception:
        return -1
    finally:
        for p in paths:
            try:
                os.remove(p.replace("file://", ""))
            except Exception:
                pass


def smolvlm_text_chat(
    prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 64,
) -> str:
    base_url = base_url or os.getenv("SMOLVLM_VLLM_URL", "http://localhost:8000/v1")
    api_key = api_key or os.getenv("SMOLVLM_VLLM_API_KEY", "test-key")
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SmolVLM2 vLLM API.")
    parser.add_argument("--image1", help="Path to first image")
    parser.add_argument("--image2", help="Path to second image")
    parser.add_argument(
        "--prompt",
        default="Which image better achieves the goal?",
        help="Text prompt to guide preference or chat",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("SMOLVLM_VLLM_URL", "http://localhost:8000/v1"),
        help="vLLM OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SMOLVLM_VLLM_API_KEY", "test-key"),
        help="API key for vLLM server",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="Max tokens for text chat mode"
    )
    args = parser.parse_args()

    # Text-only mode
    if args.image1 is None and args.image2 is None:
        reply = smolvlm_text_chat(
            args.prompt, base_url=args.base_url, api_key=args.api_key, max_tokens=args.max_tokens
        )
        print(reply)
    # Image pair preference mode
    elif args.image1 and args.image2:
        img1 = np.array(Image.open(args.image1).convert("RGB"))
        img2 = np.array(Image.open(args.image2).convert("RGB"))
        pref = smolvlm_pair_preference(
            img1, img2, args.prompt, base_url=args.base_url, api_key=args.api_key
        )
        print(f"Preference (0=img1, 1=img2, -1=unknown): {pref}")
    else:
        parser.error("Provide both --image1 and --image2 for preference, or neither for text chat.")
