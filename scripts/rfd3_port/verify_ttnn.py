"""Verify the ttnn TokenInitializer port against the torch reference + captured goldens.

Two passes (standard port-parity methodology):
  1. random-weight PCC: ttnn vs torch-ref with IDENTICAL random init (structural sanity).
  2. real-weight PCC: ttnn (real ckpt weights) vs captured golden outs (target > 0.99).

Usage:
  TT_VISIBLE_DEVICES=0 python verify_ttnn.py [capture_dir]
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import rfd3_ref as R  # vendored torch reference

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import ttnn
from tt_bio.rfd3 import build_token_initializer


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d.clamp(min=1e-12)) if d > 0 else 0.0


def load_f(capture_dir):
    f = {}
    for fn in os.listdir(capture_dir):
        if fn.startswith("token_initializer.in_f_") and fn.endswith(".pt"):
            key = fn[len("token_initializer.in_f_"):-3]
            t = torch.load(os.path.join(capture_dir, fn), map_location="cpu")
            # Golden f was captured under bf16 AMP; the torch ref runs in fp32 (only the
            # Pairformer block autocasts to bf16). Cast fp tensors to float32; leave
            # bool/int index/mask tensors untouched (matches verify_ref.py).
            if t.is_floating_point() and t.dtype != torch.float32:
                t = t.float()
            f[key] = t
    return f


def load_golden(capture_dir):
    g = {}
    for fn in os.listdir(capture_dir):
        if fn.startswith("token_initializer.out_") and fn.endswith(".pt"):
            name = fn[len("token_initializer.out_"):-3]
            g[name] = torch.load(os.path.join(capture_dir, fn), map_location="cpu")
    return g


def report(title, out, ref):
    print(f"\n=== {title} ===")
    keys = sorted(set(out) | set(ref))
    ok = True
    for k in keys:
        if k in out and k in ref:
            p = pcc(out[k], ref[k])
            mae = (out[k].float() - ref[k].float()).abs().max().item()
            print(f"  {k:12s} PCC={p:.6f}  max_abs={mae:.4e}  shape={tuple(out[k].shape)}")
            if p < 0.9:
                ok = False
        else:
            print(f"  {k:12s} MISSING (out={k in out}, ref={k in ref})")
    return ok


def main(capture_dir):
    f = load_f(capture_dir)
    golden = load_golden(capture_dir)
    print(f"[verify_ttnn] f keys={len(f)}; golden keys={sorted(golden.keys())}")

    # ---- Pass 1: random-weight PCC (ttnn vs ref, identical init) ----
    torch.manual_seed(0)
    ref = R.build_token_initializer()
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    ref.eval()
    with torch.no_grad():
        ref_out = ref(copy.deepcopy(f))
    print("[verify_ttnn] built torch ref (random weights); running ttnn port with same weights...")
    ttnn_mod = build_token_initializer(sd)
    ttnn_out = ttnn_mod(copy.deepcopy(f))
    report("random-weight PCC (ttnn vs torch-ref)", ttnn_out, ref_out)

    # ---- Pass 2: real-weight PCC (ttnn vs golden) ----
    wpath = os.path.join(capture_dir, "token_initializer.real_weights.pt")
    real_sd = torch.load(wpath, map_location="cpu")
    print(f"[verify_ttnn] loaded {len(real_sd)} real weights; running ttnn port...")
    ttnn_mod2 = build_token_initializer(real_sd)
    ttnn_out2 = ttnn_mod2(copy.deepcopy(f))
    report("real-weight PCC (ttnn vs golden)", ttnn_out2, golden)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/moritz/.coworker/wt/tt-bio-rfdiffusion3-port-p3/.scratch/rfd3-ref/goldens/capture")
