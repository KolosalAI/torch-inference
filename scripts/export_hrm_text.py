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


class PrefillWrapper(nn.Module):
    """First-pass wrapper. Takes input_ids; returns logits and a flat tuple of
    (k0, v0, k1, v1, ..., kN-1, vN-1) cache tensors so torch.export can pytree
    them."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids=input_ids, use_cache=True, return_dict=True)
        kvs = []
        for layer_kv in out.past_key_values:  # DynamicCache iterable of (k, v)
            kvs.extend([layer_kv[0], layer_kv[1]])
        return (out.logits, *kvs)


class DecodeStepWrapper(nn.Module):
    """Cached-pass wrapper. Takes one new token + the flat cache tuple from the
    previous step; returns updated logits and a fresh flat cache tuple."""
    def __init__(self, m, num_layers):
        super().__init__()
        self.m = m
        self.L = num_layers

    def forward(self, input_ids, *past_flat):
        # transformers v5 cache_utils
        from transformers.cache_utils import DynamicCache
        past = [(past_flat[2 * i], past_flat[2 * i + 1]) for i in range(self.L)]
        cache = DynamicCache.from_legacy_cache(past)
        out = self.m(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        new = []
        for layer_kv in out.past_key_values:
            new.extend([layer_kv[0], layer_kv[1]])
        return (out.logits, *new)


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


def export_two_graph(slow_loops_override, fast_loops_override):
    """Emit prefill.onnx + decode_step.onnx with explicit past/present KV I/O.

    Required artifacts:
      OUT_DIR/prefill.onnx          (graph)
      OUT_DIR/prefill.onnx.data     (external weights, ~2.3 GB)
      OUT_DIR/decode_step.onnx
      OUT_DIR/decode_step.onnx.data
      OUT_DIR/config.json (extended with num_heads/head_dim)
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_ID} (two-graph KV export)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.train(False)
    # KV path requires use_cache=True for both graphs
    model.config.use_cache = True

    hf_cfg = model.config.to_dict()
    num_layers = hf_cfg["num_hidden_layers"]   # 16
    num_heads = hf_cfg["num_attention_heads"]
    head_dim = hf_cfg["hidden_size"] // num_heads

    # --- Prefill graph ---
    prefill = PrefillWrapper(model)
    sample_ids = tok("hello world", return_tensors="pt").input_ids  # [1, n]
    prefill_path = f"{OUT_DIR}/prefill.onnx"
    print(f"Exporting prefill -> {prefill_path}...")
    pkv_output_names = []
    for i in range(num_layers):
        pkv_output_names.extend(
            [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
        )
    dyn_axes = {"input_ids": {1: "seq"}, "logits": {1: "seq"}}
    for name in pkv_output_names:
        dyn_axes[name] = {2: "seq"}
    with torch.no_grad():
        torch.onnx.export(
            prefill, (sample_ids,), prefill_path,
            opset_version=18,
            input_names=["input_ids"],
            output_names=["logits", *pkv_output_names],
            dynamic_axes=dyn_axes,
            do_constant_folding=True,
        )

    # --- Decode-step graph ---
    # Build a one-token sample input and matching synthetic cache from a real
    # prefill run so shapes line up.
    with torch.no_grad():
        out0 = model(input_ids=sample_ids, use_cache=True, return_dict=True)
    sample_step_ids = sample_ids[:, -1:].clone()  # [1, 1]
    past_flat = []
    for layer_kv in out0.past_key_values:
        past_flat.extend([layer_kv[0], layer_kv[1]])

    decode = DecodeStepWrapper(model, num_layers)
    decode_path = f"{OUT_DIR}/decode_step.onnx"
    print(f"Exporting decode_step -> {decode_path}...")
    pkv_input_names = []
    present_output_names = []
    for i in range(num_layers):
        pkv_input_names.extend(
            [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
        )
        present_output_names.extend(
            [f"present_key_values.{i}.key", f"present_key_values.{i}.value"]
        )
    dyn_axes = {
        "input_ids": {1: "one"},
        "logits": {1: "one"},
    }
    for name in pkv_input_names:
        dyn_axes[name] = {2: "past_len"}
    for name in present_output_names:
        dyn_axes[name] = {2: "past_len_plus_one"}
    with torch.no_grad():
        torch.onnx.export(
            decode, (sample_step_ids, *past_flat), decode_path,
            opset_version=18,
            input_names=["input_ids", *pkv_input_names],
            output_names=["logits", *present_output_names],
            dynamic_axes=dyn_axes,
            do_constant_folding=True,
        )

    # tokenizer.json + extended config.json
    tok.save_pretrained(OUT_DIR)
    runtime_cfg = {
        "eos_token_id": hf_cfg.get("eos_token_id"),
        "ctx_size": hf_cfg.get("max_position_embeddings", 2048),
        "slow_loops": slow_loops_override or hf_cfg.get("H_cycles", 2),
        "fast_loops": fast_loops_override or hf_cfg.get("L_cycles", 3),
        "vocab_size": hf_cfg.get("vocab_size"),
        "hidden_size": hf_cfg.get("hidden_size"),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "logits_dtype": "float16",
    }
    with open(f"{OUT_DIR}/config.json", "w") as f:
        json.dump(runtime_cfg, f, indent=2)
    print("Two-graph export done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true", help="also emit an int8 variant")
    ap.add_argument("--slow-loops", type=int, default=None,
                    help="override H_cycles in runtime config (default: from checkpoint)")
    ap.add_argument("--fast-loops", type=int, default=None,
                    help="override L_cycles in runtime config (default: from checkpoint)")
    ap.add_argument("--two-graph", action="store_true",
                    help="emit prefill.onnx + decode_step.onnx with KV-cache I/O")
    args = ap.parse_args()
    if args.two_graph:
        export_two_graph(args.slow_loops, args.fast_loops)
    else:
        export(args.quantize, args.slow_loops, args.fast_loops)
