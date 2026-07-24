"""Verify the vendored torch reference reproduces the captured golden outputs with
the real weights (real-weight reference PCC; expect ~1.0). Run locally with the
shared env (torch+einops+numpy):

    python verify_ref.py /path/to/capture_dir

Loads token_initializer.in_f_*.pt + token_initializer.real_weights.pt +
token_initializer.out_*.pt, builds the ref, loads weights, runs forward, prints
per-output PCC + max-abs.
"""
import json
import os
import sys

import torch

import rfd3_ref as R


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom.clamp(min=1e-12)).item() if denom > 0 else 0.0


def main(capture_dir):
    # Load f dict
    f = {}
    for fn in os.listdir(capture_dir):
        if fn.startswith("token_initializer.in_f_") and fn.endswith(".pt"):
            key = fn[len("token_initializer.in_f_"):-3]
            t = torch.load(os.path.join(capture_dir, fn), map_location="cpu")
            # Golden f was captured under bf16 AMP; run the fp32 reference in float32
            # (PCC vs bf16 golden is unaffected by the fp32-vs-bf16 rounding). Only cast
            # floating-point tensors; leave bool/int index/mask tensors untouched.
            if t.is_floating_point() and t.dtype != torch.float32:
                t = t.float()
            f[key] = t
    print(f"[verify_ref] loaded f dict with {len(f)} keys: {sorted(f.keys())}")
    # Load real weights
    wpath = os.path.join(capture_dir, "token_initializer.real_weights.pt")
    sd = torch.load(wpath, map_location="cpu")
    print(f"[verify_ref] loaded {len(sd)} real weight tensors")
    # Build ref + load
    m = R.build_token_initializer()
    msd = m.state_dict()
    # Sanity: every real weight key should be in msd
    missing = [k for k in sd if k not in msd]
    extra = [k for k in msd if k not in sd]
    print(f"[verify_ref] weight key check: missing_in_ref={len(missing)} extra_in_ref={len(extra)}")
    if missing:
        print("  missing (in ckpt, not in ref):", missing[:10])
    if extra:
        print("  extra (in ref, not in ckpt):", extra[:10])
    msd.update(sd)
    m.load_state_dict(msd, strict=False)
    m.eval()
    # Load golden outs
    golden = {}
    for fn in os.listdir(capture_dir):
        if fn.startswith("token_initializer.out_") and fn.endswith(".pt"):
            name = fn[len("token_initializer.out_"):-3]
            golden[name] = torch.load(os.path.join(capture_dir, fn), map_location="cpu")
    print(f"[verify_ref] golden outs: { {k: tuple(v.shape) for k,v in golden.items()} }")
    with torch.no_grad():
        out = m(f)
    print("[verify_ref] ref outs:", {k: tuple(v.shape) for k, v in out.items()})
    print("\n=== real-weight reference PCC (ref vs golden) ===")
    for name in sorted(set(out) | set(golden)):
        if name in out and name in golden:
            p = pcc(out[name], golden[name])
            mae = (out[name].float() - golden[name].float()).abs().max().item()
            print(f"  {name:12s} PCC={p:.6f}  max_abs_err={mae:.6e}  shape={tuple(out[name].shape)}")
        else:
            print(f"  {name:12s} MISSING (out={name in out}, golden={name in golden})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/work/capture")
