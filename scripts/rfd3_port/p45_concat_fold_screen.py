"""Lever A screen -- can the 258-wide concat chain be replaced by two 128-wide gathers?

S2 put the chain at 165.7 ms/step of the token encoder, all of it at a fraction of the roof:

    concat[z, onehot_d, onehot_s] -> 99.5    rms_norm(258) -> 6.8    linear 258->128 -> 25.6
    the two 65-wide embeddings + their to_layout -> 33.8

The identity this screens. `rms_norm` scales a row by r = 1/sqrt(mean(x^2) + eps), and each one-hot
row contributes exactly one 1.0, so

    mean(x^2) = (sum_c z_c^2 + 1 + 1) / 258

is computable from z alone -- the one-hot values never enter it. And `onehot(b) @ W_d` is row b of
W_d, so the 65-wide one-hot never has to exist: fold the per-channel rms weight into the table once
(Wd_pre[b] = w[128+b] * W_d[b]) and the whole block becomes

    out = ((z * r) * w_z) @ W_z  +  r * (Wd_pre[bins_d] + Ws_pre[bins_s])

`Z_init_II` is invariant across timesteps and recycles (model.py:2830 passes the same `Z_II` into
every `_process_`, which is why `_tt_cached` already hits on its upload), and so is `r`. So the
first term is a CONSTANT of the design: computed once, reused for all 200 x 2 calls.

This screen answers only the first question -- is the algebra exact? -- in float64, where the answer
is data-independent, plus a bf16 round-trip for a first read on the rounding. **The real gate is a
200-step device trajectory, and this does not substitute for it.**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

N_BINS = 65
C_Z = 128
EPS = 1e-6


def shipped(z, oh_d, oh_s, w, W, dt):
    """The chain as it runs today: concat -> rms_norm(258) -> linear(258->128)."""
    x = torch.cat([z, oh_d, oh_s], dim=-1).to(dt)
    r = torch.rsqrt(x.to(dt).pow(2).mean(-1, keepdim=True) + EPS)
    return ((x * r) * w.to(dt)) @ W.to(dt).T


def folded(z, bins_d, bins_s, w, W, dt):
    """The gather-fold. Same maths, no 258-wide tensor and no 65-wide one-hot."""
    n = z.shape[-1] + 2 * N_BINS
    r = torch.rsqrt((z.to(dt).pow(2).sum(-1, keepdim=True) + 2.0) / n + EPS)
    Wz = W[:, :C_Z]                       # [128, 128]
    Wd = W[:, C_Z:C_Z + N_BINS]           # [128, 65]
    Ws = W[:, C_Z + N_BINS:]              # [128, 65]
    wz, wd, ws = w[:C_Z], w[C_Z:C_Z + N_BINS], w[C_Z + N_BINS:]
    # fold the per-channel rms weight into the tables once: [65, 128] each
    Wd_pre = (Wd * wd).T.to(dt)
    Ws_pre = (Ws * ws).T.to(dt)
    const = ((z.to(dt) * r) * wz.to(dt)) @ Wz.to(dt).T   # step-invariant
    return const + r * (Wd_pre[bins_d] + Ws_pre[bins_s])


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--tokens", type=int, default=96, help="I; the identity is size-independent")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    sd = torch.load(Path(a.ckpt) / "diffusion_module.real_weights.pt", map_location="cpu",
                    weights_only=True)
    w = sd["diffusion_token_encoder.process_z.0.weight"].double()
    W = sd["diffusion_token_encoder.process_z.1.weight"].double()
    B, I = a.batch, a.tokens
    g = torch.Generator().manual_seed(7)
    z = torch.randn(B, I, I, C_Z, generator=g).double() * 0.5
    bins_d = torch.randint(0, N_BINS, (B, I, I), generator=g)
    bins_s = torch.randint(0, N_BINS, (B, I, I), generator=g)
    oh_d = torch.nn.functional.one_hot(bins_d, N_BINS).double()
    oh_s = torch.nn.functional.one_hot(bins_s, N_BINS).double()

    res = {}
    for tag, dt in (("float64", torch.float64), ("float32", torch.float32),
                    ("bfloat16", torch.bfloat16)):
        ref = shipped(z, oh_d, oh_s, w, W, dt).double()
        got = folded(z, bins_d, bins_s, w, W, dt).double()
        d = (ref - got).abs()
        rel = float(d.max() / ref.abs().max())
        res[tag] = {"pcc": pcc(ref, got), "max_abs": float(d.max()), "max_rel": rel}
        print(f"{tag:9s} PCC {res[tag]['pcc']:.10f}   max|d| {float(d.max()):.3e}   "
              f"max rel {rel:.3e}", flush=True)

    print(f"\n[algebra] the identity is {'EXACT' if res['float64']['max_rel'] < 1e-12 else 'WRONG'} "
          f"in float64 -- data-independent, so this settles the maths.", flush=True)
    print("[rounding] the bfloat16 row is a first read only. The gate is a 200-step device "
          "trajectory against the shipped path at the same seed.", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"tokens": I, "batch": B, "results": res}, indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
