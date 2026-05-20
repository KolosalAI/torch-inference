"""Parity test: run 64-token greedy decode via the existing monolithic
model.onnx AND via the new prefill.onnx + decode_step.onnx. Assert
identical token IDs. Run AFTER `python scripts/export_hrm_text.py --two-graph`.

Exits non-zero on mismatch. Intended to gate the Rust work in Phase 3.
"""
import json
import sys
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

OUT_DIR = "services/llm/models/hrm-text-1b"
PROMPT = "The capital of France is"
N_TOKENS = 64


def greedy_monolithic(sess, ids):
    out = []
    for _ in range(N_TOKENS):
        logits = sess.run(["logits"], {"input_ids": ids})[0]
        next_id = int(np.argmax(logits[0, -1]))
        out.append(next_id)
        ids = np.concatenate([ids, [[next_id]]], axis=1).astype(np.int64)
    return out


def greedy_kv(prefill_sess, decode_sess, ids, cfg):
    L = cfg["num_layers"]
    # Prefill
    pkv_out_names = ["logits"] + [
        f"past_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")
    ]
    prefill_outs = prefill_sess.run(pkv_out_names, {"input_ids": ids})
    logits = prefill_outs[0]
    past = prefill_outs[1:]  # 2*L tensors, alternating k,v
    next_id = int(np.argmax(logits[0, -1]))
    out = [next_id]

    # Decode loop
    pkv_in_names = ["input_ids"] + [
        f"past_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")
    ]
    present_names = ["logits"] + [
        f"present_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")
    ]
    for _ in range(N_TOKENS - 1):
        inputs = {"input_ids": np.array([[next_id]], dtype=np.int64)}
        for n, t in zip(pkv_in_names[1:], past):
            inputs[n] = t
        outs = decode_sess.run(present_names, inputs)
        logits = outs[0]
        past = outs[1:]
        next_id = int(np.argmax(logits[0, -1]))
        out.append(next_id)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(OUT_DIR)
    ids = tok(PROMPT, return_tensors="np").input_ids.astype(np.int64)

    with open(f"{OUT_DIR}/config.json") as f:
        cfg = json.load(f)

    print("Running monolithic decode...")
    mono = ort.InferenceSession(f"{OUT_DIR}/model.onnx",
                                providers=["CPUExecutionProvider"])
    mono_tokens = greedy_monolithic(mono, ids.copy())

    print("Running two-graph KV decode...")
    pref = ort.InferenceSession(f"{OUT_DIR}/prefill.onnx",
                                providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{OUT_DIR}/decode_step.onnx",
                                providers=["CPUExecutionProvider"])
    kv_tokens = greedy_kv(pref, dec, ids.copy(), cfg)

    if mono_tokens != kv_tokens:
        print(f"MISMATCH after {next(i for i, (a, b) in enumerate(zip(mono_tokens, kv_tokens)) if a != b)} tokens")
        print(f"  monolithic: {mono_tokens}")
        print(f"  kv-cache:   {kv_tokens}")
        sys.exit(1)
    print(f"PARITY OK — {N_TOKENS} tokens identical: {mono_tokens[:8]}...")


if __name__ == "__main__":
    main()
