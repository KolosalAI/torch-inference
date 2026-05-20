"""
HRM-Text -> ONNX exporter (one-time, offline, build-time only).
Output: services/llm/models/hrm-text-1b/{model.onnx, model.onnx.data, tokenizer.json, config.json}

Run via:  make hrm-export

Requires: Python 3.10+, transformers v5 (from git), onnxscript.
The hrm-export Makefile target provisions these.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sapientinc/HRM-Text-1B"
OUT_DIR = "services/llm/models/hrm-text-1b"


class LogitsOnly(nn.Module):
    """Wraps the HF model so torch.export sees a single Tensor output instead of
    a CausalLMOutputWithPast that contains a DynamicCache (torch can't pytree-
    flatten DynamicCache). Required for export to succeed."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids=input_ids, use_cache=False, return_dict=True)
        return out.logits


def export(quantize: bool, slow_loops_override, fast_loops_override):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.train(False)
    model.config.use_cache = False  # belt-and-braces; matches LogitsOnly wrapper

    wrapped = LogitsOnly(model)
    ids = tok("hello", return_tensors="pt").input_ids
    onnx_path = f"{OUT_DIR}/model.onnx"

    print("Exporting ONNX (opset 18 - torch auto-upgrades from 17)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped, (ids,), onnx_path,
            opset_version=18,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
            do_constant_folding=True,
        )
    # Export emits model.onnx (~32 MB graph) + model.onnx.data (~2.2 GB external weights).
    # Both files MUST be deployed together in the same directory.
    print(f"  -> {onnx_path} (+ model.onnx.data sidecar)")

    # tokenizer.json is what the Rust runtime reads
    tok.save_pretrained(OUT_DIR)

    hf_cfg = model.config.to_dict()
    runtime_cfg = {
        "eos_token_id": hf_cfg.get("eos_token_id"),
        "ctx_size": hf_cfg.get("max_position_embeddings", 2048),
        # Loop counts: documentation only. The recurrence is unrolled in the
        # static graph; the Rust runtime does not loop on these.
        "slow_loops": slow_loops_override or hf_cfg.get("H_cycles", 2),
        "fast_loops": fast_loops_override or hf_cfg.get("L_cycles", 3),
        "vocab_size": hf_cfg.get("vocab_size"),
        "hidden_size": hf_cfg.get("hidden_size"),
        "num_layers": hf_cfg.get("num_hidden_layers"),
        "logits_dtype": "float16",  # matches torch_dtype above
    }
    with open(f"{OUT_DIR}/config.json", "w") as f:
        json.dump(runtime_cfg, f, indent=2)
    print(f"  -> {OUT_DIR}/config.json")

    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q_path = f"{OUT_DIR}/model.int8.onnx"
        quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
        print(f"  -> {q_path} (int8)")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true", help="also emit an int8 variant")
    ap.add_argument("--slow-loops", type=int, default=None,
                    help="override H_cycles in runtime config (default: from checkpoint)")
    ap.add_argument("--fast-loops", type=int, default=None,
                    help="override L_cycles in runtime config (default: from checkpoint)")
    args = ap.parse_args()
    export(args.quantize, args.slow_loops, args.fast_loops)
