"""
One-time conversion: kokoro-v1_0.pth -> kokoro-v1_0.safetensors
Run before first use of the torch TTS engine.
Requires: pip install torch safetensors
Usage: python convert_kokoro.py
"""
import torch
from safetensors.torch import save_file
import sys
import os

src = "models/kokoro-82m/kokoro-v1_0.pth"
dst = "models/kokoro-82m/kokoro-v1_0.safetensors"

if not os.path.exists(src):
    print(f"ERROR: {src} not found. Download from:")
    print("  https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth")
    sys.exit(1)

print(f"Loading {src}...")
state = torch.load(src, map_location="cpu", weights_only=True)
if not isinstance(state, dict):
    state = state.state_dict()

# safetensors requires contiguous tensors
state = {k: v.contiguous() for k, v in state.items()}
os.makedirs(os.path.dirname(dst), exist_ok=True)
save_file(state, dst)
print(f"Saved {dst} ({len(state)} tensors)")
print("First 10 keys:", list(state.keys())[:10])
print("Done. Now build with: cargo build --features torch")
