"""Which pair tensors can be stored bfp8_b, card-free: the float64 RFD4 trunk with a tensor rounded the way the packer
stores it, graded against float64 (bar 0.05 rel RMS, as zprec.py and the golden test).

    python perf/rfd4/z8/zprec_cpu.py --out <json> [--n 128] [--cycles 1] [--check-ttnn]

Arms, each over the same golden inputs tiled to `--n` tokens (zprec.py's construction):
  main   bf16 at every op boundary, s fp32, z bf16 (pair_bf16z): what RFD4 runs today
  x8     main + the triangle contraction's output x stored bfp8 (16-value blocks along j, its [C, S, S] tile rows)
  z8     main + the pair stream z stored bfp8 at every write (16-value blocks along the channels, z's tile rows)
  x8z8   both
`--check-ttnn` first checks `bfp8` against ttnn's own host conversion (no device is opened).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import torch  # noqa: E402

import rfd4_reference as REF  # noqa: E402

GOLDENS = Path.home() / "rfd4-goldens"
HEADS = 12
BAR = 5e-2


def bfp8(t, dim=-1, trunc=False, pre=None, eblock=16):
    """bfp8_b as Tensix stores it: 16 consecutive values along `dim` share the largest one's exponent; each keeps a
    sign and 7 magnitude bits at that scale (a value 2^k below the block's largest loses k bits)."""
    t = t.movedim(dim, -1)
    shp = t.shape
    g = t.float().reshape(*shp[:-1], shp[-1] // 16, 16)
    if eblock != 16:      # round at the step a wider block's max implies (e.g. a 32-wide FPU row max), then truncate
        gw = t.float().reshape(*shp[:-1], shp[-1] // eblock, eblock)
        ew = torch.frexp(gw.abs().amax(-1, keepdim=True))[1] - 1
        half = torch.ldexp(torch.ones_like(gw), (ew - 7).expand_as(gw))
        g = (gw + torch.sign(gw) * half).reshape(g.shape)
    if pre is not None:   # a per-element pre-round before the packer truncates: + sign * 2^(own exponent - 7 + pre)
        g = g + torch.sign(g) * torch.ldexp(torch.ones_like(g), torch.frexp(g)[1] - 1 - 7 + pre)
    e = torch.frexp(g.abs().amax(-1, keepdim=True))[1] - 1            # floor(log2(max)), 0 for an all-zero block
    step = torch.ldexp(torch.ones_like(g[..., :1]), e - 6)
    q = g / step
    q = (torch.trunc(q) if trunc else torch.round(q)).clamp(-127, 127)
    return (q * step).double().reshape(shp).movedim(-1, dim)


def mode(x8, z8, trunc=False, pre=None, eblock=16):
    def rnd(t):
        return REF.bf16(t)
    rnd.residual = REF.f32          # s stays fp32
    rnd.z = (lambda t: bfp8(t, -1, trunc, pre, eblock)) if z8 else REF.bf16
    if x8:
        rnd.x = lambda t: bfp8(t, 2, trunc)   # x [b, i, j, c]; the card's x tile rows run along j
    return rnd


def tile(t, n, dims):
    for d in dims:
        reps = [1] * t.dim()
        reps[d] = -(-n // t.shape[d])
        t = t.repeat(*reps).narrow(d, 0, n)
    return t


def check_ttnn():
    import ttnn
    torch.manual_seed(0)
    t = torch.randn(64, 64) * torch.logspace(-3, 1, 64)[None, :]
    ref = ttnn.to_torch(ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).double()
    out = {}
    for name, tr in (("round", False), ("trunc", True)):
        mine = bfp8(t.bfloat16().double(), -1, tr)
        out[name] = {"max_abs_diff": (mine - ref).abs().max().item(), "equal_frac": (mine == ref).double().mean().item()}
    print("ttnn host bfp8 vs emulation", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--trunc", action="store_true", help="truncate instead of round to nearest (an upper bound)")
    ap.add_argument("--pre", type=int, default=None, help="per-element pre-round offset (with --trunc)")
    ap.add_argument("--eblock", type=int, default=16, help="with --trunc: pre-round at the step of this many values' max")
    ap.add_argument("--check-ttnn", action="store_true")
    ap.add_argument("--arms", default="main,x8,z8,x8z8")
    a = ap.parse_args()
    res = {"tokens": a.n, "cycles": a.cycles, "bar": BAR, "trunc": a.trunc, "pre": a.pre, "eblock": a.eblock}
    if a.check_ttnn:
        res["ttnn_check"] = check_ttnn()
    weights = {k: v.double() for k, v in torch.load(GOLDENS / "trunk-weights" / "flow_trunk.pt", map_location="cpu",
                                                     weights_only=False)["weights"].items()}
    d = torch.load(GOLDENS / "unconditional_128" / "MultiheadAttnAndTransition__nn_transformer_layers_1__call0.pt",
                   map_location="cpu", weights_only=False)
    x, z, cond, mask = [t.double() for t in d["args"]][:4]
    m = mask[0].bool()
    x, z, cond = x[:, m], z[:, m][:, :, m], cond[:, m]
    n = a.n
    x, z, cond = tile(x, n, [1]), tile(z, n, [1, 2]), tile(cond, n, [1])
    mask = torch.ones(1, n, dtype=torch.float64)
    rel = lambda g, r: ((g - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()  # noqa: E731
    t0 = time.time()
    ref_s, ref_z = REF.trunk(weights, x, z, [cond] * a.cycles, mask, 14, n_heads=HEADS)
    print(f"float64 {time.time() - t0:.0f} s", flush=True)
    wb = {k: REF.bf16(v) for k, v in weights.items()}
    outs = {}
    arms = {"main": (False, False), "x8": (True, False), "z8": (False, True), "x8z8": (True, True)}
    for name in a.arms.split(","):
        x8, z8 = arms[name]
        s_, z_ = REF.trunk(wb, x, z, [cond] * a.cycles, mask, 14, n_heads=HEADS, rnd=mode(x8, z8, a.trunc, a.pre, a.eblock))
        outs[name] = (s_, z_)
        res[name] = {"s": rel(s_, ref_s), "z": rel(z_, ref_z)}
        if "main" in outs and name != "main":
            res[name]["vs_main"] = {"s": rel(s_, outs["main"][0]), "z": rel(z_, outs["main"][1])}
        print(name, res[name], f"{time.time() - t0:.0f} s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
