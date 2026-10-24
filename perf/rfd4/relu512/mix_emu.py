"""Card-free: every per-matmul fidelity mix of `pair_relu` (fc1 384->768 + ReLU, fc2 768->384) against float64.

    python -I perf/rfd4/relu512/mix_emu.py --golden <PairReprUpdate golden .pt> --weights <model.safetensors> \
        --layer 12 --out <json>

Each matmul is emulated as the FPU's phase sum. srcA keeps its hidden bit + 4 MSBs (A5) in phase 1 and its next 5 bits
in phase 2; srcB keeps its hidden bit + 6 MSBs (B7), the rest only in HiFi3/4:
    LoFi  = A5 * B7            HiFi2 = A10 * B7            HiFi4 = A * B
Per matmul a mix picks the fidelity, which operand sits in srcA ("wA": the weight, as compute.cpp does today; "wB": the
operands swapped, the transposed product) and the weight format (bfp8 or bf16). Accumulation is float64 (fp32 in dest
on the chip), so each arm's error is the operand truncation alone. Everything else is the kernel's: LayerNorm in float,
bf16 normed input, bias in bf16, bf16 hidden between the matmuls, bf16 output.
"""
import argparse
import itertools
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
    """x's significand (hidden bit included) truncated toward zero to `bits` bits."""
    m, e = torch.frexp(x)
    s = 2.0 ** bits
    return torch.ldexp(torch.trunc(m * s) / s, e)


def bf16(x):
    return x.to(torch.bfloat16).double()


def bfp8(w):
    """bfloat8_b of [K, N]: 16 consecutive values of a face row share the largest exponent, 7 magnitude bits."""
    k, n = w.shape
    b = w.reshape(k, n // 16, 16)
    _, e = torch.frexp(b.abs().amax(-1, keepdim=True))
    q = 2.0 ** (e - 7)
    return (torch.round(b / q).clamp(-127, 127) * q).reshape(k, n)


SRC_A = {"LoFi": 5, "HiFi2": 10, "HiFi4": None}
SRC_B = {"LoFi": 7, "HiFi2": 7, "HiFi4": None}


def mm(x, w, fid, side):
    """x @ w as the FPU computes it at `fid` with the weight in srcA (side "wA") or srcB ("wB")."""
    ka, kb = SRC_A[fid], SRC_B[fid]
    t = lambda v, k: v if k is None else trunc(v, k)  # noqa: E731
    if side == "wA":
        return t(x, kb) @ t(w, ka)
    return t(x, ka) @ t(w, kb)


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
    fmt = {"bfp8": (bfp8(w1), bfp8(w2)), "bf16": (bf16(w1), bf16(w2))}
    choices = [("HiFi4", "wA"), ("HiFi2", "wA"), ("HiFi2", "wB"), ("LoFi", "wA"), ("LoFi", "wB")]
    rows = []
    for wf, c1, c2 in itertools.product(fmt, choices, choices):
        q1, q2 = fmt[wf]
        h = bf16(torch.relu(mm(bf16(xn), q1, *c1) + bf16(b1)))
        y = bf16(mm(h, q2, *c2) + bf16(b2))
        r = dict(weights=wf, fc1="%s/%s" % c1, fc2="%s/%s" % c2, rel_rms=rel(y))
        rows.append(r)
        print(r, flush=True)
    out = dict(layer=a.layer, rows=a.rows, mixes=rows, x_std=float(x.std()), golden=a.golden,
               script="perf/rfd4/relu512/mix_emu.py")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
