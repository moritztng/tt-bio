#!/usr/bin/env python3
"""Scout-only: run the tt-bio hot ttnn op mix on one card and dump outputs + warm timings.

Same script, same seed, same inputs under each venv, so outputs are directly comparable:
"did 0.75 move the numbers" is answered by diffing two JSONs, not by opinion.

Usage: TT_VISIBLE_DEVICES=1 python3 scripts/scout_device_probe.py <out.json> [reps]
"""
import json, sys, time
import torch
import ttnn

OUT = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
res = {"ops": {}}
import importlib.metadata as md
res["ttnn_version"] = md.version("ttnn")


def sync():
    ttnn.synchronize_device(dev)


def record(name, fn):
    """fn() re-runs the op and returns a fresh device tensor."""
    host = ttnn.to_torch(fn()).float()
    res["ops"][name] = {
        "shape": list(host.shape),
        "mean": float(host.mean()), "std": float(host.std()),
        "min": float(host.min()), "max": float(host.max()),
        "first256": [round(float(v), 6) for v in host.flatten()[:256]],
    }
    for _ in range(3):          # warm the kernel cache before any timing
        fn()
    sync()                      # drain, so the timed region starts empty
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    sync()                      # drain, so queued device time is inside the region
    res["ops"][name]["ms_per_rep"] = (time.perf_counter() - t0) * 1e3 / REPS
    print("  %-28s %8.3f ms" % (name, res["ops"][name]["ms_per_rep"]), flush=True)


def tt(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=dev)


PROBES = []


def probe(name):
    def deco(f):
        PROBES.append((name, f))
        return f
    return deco


# --- matmul (trunk + RFD3 atom blocks) ---
a = tt(torch.randn(1, 1, 1024, 512)); b = tt(torch.randn(1, 1, 512, 512))
probe("matmul_1024x512x512")(lambda: ttnn.matmul(a, b))

# --- norms (ESMC, trunk) ---
x = tt(torch.randn(1, 1, 1024, 512)); w = tt(torch.randn(512)); bi = tt(torch.randn(512))
probe("layer_norm_1024x512")(lambda: ttnn.layer_norm(x, weight=w, bias=bi))
probe("rms_norm_1024x512")(lambda: ttnn.rms_norm(x, weight=w))

# --- softmax (attention; exposed to the bf16 accurate-exp re-land) ---
s = tt(torch.randn(1, 8, 512, 512))
probe("softmax_8x512x512")(lambda: ttnn.softmax(s, dim=-1))

# --- SDPA (most at risk from the SFPI rounding-mode + accurate-exp changes) ---
q = tt(torch.randn(1, 8, 512, 64)); k = tt(torch.randn(1, 8, 512, 64)); v = tt(torch.randn(1, 8, 512, 64))
probe("sdpa_8h_512_64")(lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False))

# --- concat at non-tile-aligned width (the known cliff; PR 46321 chunking) ---
c1 = tt(torch.randn(1, 1, 256, 48)); c2 = tt(torch.randn(1, 1, 256, 48))
probe("concat_w48_unaligned")(lambda: ttnn.concat([c1, c2], dim=3))
d1 = tt(torch.randn(1, 1, 256, 64)); d2 = tt(torch.randn(1, 1, 256, 64))
probe("concat_w64_aligned")(lambda: ttnn.concat([d1, d2], dim=3))

# --- activations touched by the SFPU accuracy PRs ---
e = tt(torch.randn(1, 1, 512, 512))
for _nm, _op in (("sigmoid", ttnn.sigmoid), ("silu", ttnn.silu), ("gelu", ttnn.gelu), ("relu", ttnn.relu)):
    probe("elt_" + _nm)(lambda o=_op: o(e))

# --- data movement ---
p = tt(torch.randn(1, 4, 256, 128))
probe("permute_0213")(lambda: ttnn.permute(p, (0, 2, 1, 3)))
probe("typecast_bf16_fp32")(lambda: ttnn.typecast(p, ttnn.float32))

for _name, _fn in PROBES:
    try:
        record(_name, _fn)
    except Exception:
        import traceback
        res["ops"][_name] = {"error": traceback.format_exc()}
        print("  %-28s FAILED" % _name, flush=True)
        print(res["ops"][_name]["error"], flush=True)

ttnn.close_device(dev)
json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
failed = [n for n, d in res["ops"].items() if "error" in d]
print("wrote %s | ops=%d | failed=%d %s" % (OUT, len(res["ops"]), len(failed), failed))
