"""Card-free: what HiFi4 / HiFi2 / LoFi drop in `pair_relu`'s two matmuls, emulated on the CPU against float64.

    python -I perf/rfd4/relu/fidelity_emu.py --golden <PairReprUpdate golden .pt> --weights <model.safetensors> \
        [--layer 12] [--rows 16384] --out <json>

In the kernel srcA holds the bfp8 weights and srcB the bf16 activation (compute.cpp: `reconfig_data_format(cb_w12,
cb_x)`, matmul_block(in0 = x, in1 = w)). The FPU multiplies srcA's hidden bit + 4 MSBs (phase 1) and its next 5 bits
(phase 2) by srcB's hidden bit + 6 MSBs; HiFi3/HiFi4 add srcB's low bits. So:
  HiFi4: every bit of both operands (bf16 x bfp8 is exact).
  HiFi2: srcB truncated to 7 significant bits: the bf16 activation loses its last mantissa bit; the weight is exact.
  LoFi:  srcB as HiFi2 and srcA truncated to 5 significant bits: the bfp8 weight loses up to 2 bits.
Accumulation is float64 here (fp32 in dest on the chip), so each arm's error is the operand rounding alone.
x is the pair entering PairReprUpdate `--layer` of a real 512-token unconditional design (the transition sees the
same stream after the write and the two triangle updates).
"""
import argparse
import json

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--golden", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--rows", type=int, default=16384)
ap.add_argument("--out", required=True)
a = ap.parse_args()
EPS = 1e-5


def trunc(x, bits):
    """Truncate x's significand (hidden bit included) to `bits` bits, toward zero, as an FPU phase drops low bits."""
    m, e = torch.frexp(x)                       # x = m * 2^e, 0.5 <= |m| < 1
    s = 2.0 ** bits
    return torch.ldexp(torch.trunc(m * s) / s, e)


def bf16(x):
    return x.to(torch.bfloat16).double()


def bfp8(w):
    """bfloat8_b of a [K, N] matrix: 16 consecutive values of a face row share the largest exponent, 7 magnitude bits
    with the hidden bit explicit, round to nearest."""
    k, n = w.shape
    b = w.reshape(k, n // 16, 16)
    _, e = torch.frexp(b.abs().amax(-1, keepdim=True))
    q = 2.0 ** (e - 7)                          # 7 bits below the shared exponent
    return (torch.round(b / q) .clamp(-127, 127) * q).reshape(k, n)


def mm(x, w, fid):
    if fid == "f64":
        return x @ w
    if fid in ("HiFi2", "LoFi"):
        x = trunc(x, 7)
    if fid == "LoFi":
        w = trunc(w, 5)
    return x @ w


def main():
    g = torch.load(a.golden, map_location="cpu", weights_only=False)
    args = g["args"] if isinstance(g, dict) and "args" in g else g
    z = next(t for t in (args if isinstance(args, (list, tuple)) else args.values())
             if torch.is_tensor(t) and t.dim() == 4 and t.shape[-1] == 384)
    x = z.reshape(-1, 384)
    x = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(0))[:a.rows]].double()
    f = safe_open(a.weights, "pt")
    p = f"nn.pair_update_layers.{a.layer}.transition_out."
    W = {k: f.get_tensor(p + k).double() for k in ("layer_norm.weight", "layer_norm.bias", "linear_1.weight",
                                                    "linear_1.bias", "linear_2.weight", "linear_2.bias")}
    w1, b1, w2, b2 = W["linear_1.weight"].t(), W["linear_1.bias"], W["linear_2.weight"].t(), W["linear_2.bias"]
    xn = torch.nn.functional.layer_norm(x, (384,), W["layer_norm.weight"], W["layer_norm.bias"], EPS)
    ref = torch.relu(xn @ w1 + b1) @ w2 + b2
    rel = lambda y: float(((y - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())  # noqa: E731
    q1, q2 = bfp8(w1), bfp8(w2)
    rows = {}
    for fid in ("f64", "HiFi4", "HiFi2", "LoFi"):
        if fid == "f64":       # bf16 activations, float weights: the floor bfp8 adds to
            h = bf16(torch.relu(mm(bf16(xn), w1, fid) + b1))
            y = mm(h, w2, fid) + b2
        else:
            h = bf16(torch.relu(mm(bf16(xn), q1, fid) + bf16(b1)))
            y = mm(h, q2, fid) + bf16(b2)
        rows["bf16_act_f64_w" if fid == "f64" else fid] = rel(bf16(y))
        print(fid, rows, flush=True)
    out = dict(layer=a.layer, rows=a.rows, rel_rms_vs_f64=rows, x_std=float(x.std()), golden=a.golden,
               script="perf/rfd4/relu/fidelity_emu.py")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
