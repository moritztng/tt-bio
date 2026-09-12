#!/usr/bin/env python3
"""Fused vs split activation at every shape a 512 aa Boltz-2 fold actually runs, plus the sweep
that turns the result into a rule instead of a list.

`b2z2-unfused-silu-recover` measured ONE point: `ttnn.linear(activation="silu")` at the pair
Transition chunk (M=16x512, K=128, N=512), 0.11432 ms fused against 0.02663 ms for the bare matmul,
and 1.4549x for the chunk once silu is split out. The census (`census.py`) says a fold runs that
call site at four different shapes and two other sites at two other activations, so the shipped
`TT_BIO_UNFUSED_SILU` flag already flips shapes nobody measured.

The mechanism predicts the rule. A fused activation is an SFPU pass over the OUTPUT, so it costs
~c_act per output tile regardless of K. The matmul that produced the output costs ~c_mm * K per
output tile. Splitting pays an L1 round trip of the output, also per output tile. So the fused
penalty as a FRACTION of the op falls like 1/K, and the win should die at a K threshold that
depends on the activation, not on the model, the site or the output size. That is the prediction
this file is built to falsify: arms at fixed output shape across K = 32..1536, four activations.

Arms, per spec, interleaved inside one process so they share a device and a program cache:

  fused    ttnn.linear(x, w, activation=act)                  what ships
  fused2   the same op again                                  the A/A floor, same rep
  split    ttnn.linear(x, w) then ttnn.<act>(y, out=y)        the candidate
  plain    ttnn.linear(x, w)                                  decomposition: what the activation costs

Accuracy is scored in the same pass against an fp32 evaluation of the identical algebra, because
the split arm rounds the pre-activation to bf16 and the fused arm does not.

    sweep.py --out <json> [--reps 9] [--inner 32] [--only census|ksweep]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--inner", type=int, default=32)
ap.add_argument("--only", default="all")
a = ap.parse_args()

import torch                                                                   # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                    # noqa: E402
import tt_bio.tenstorrent as T                                                 # noqa: E402
import tt_bio as _TB                                                           # noqa: E402
assert Path(_TB.__file__).resolve().is_relative_to(ROOT), _TB.__file__
from tt_bio.af2 import compute_kernel_config                                   # noqa: E402

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG

ACT = {"silu": ttnn.silu, "sigmoid": ttnn.sigmoid, "relu": ttnn.relu, "gelu": ttnn.gelu}
REF = {"silu": torch.nn.functional.silu, "sigmoid": torch.sigmoid,
       "relu": torch.relu, "gelu": torch.nn.functional.gelu}

# (name, in_shape, K, N, activation, bias, out_memcfg, calls_per_fold, note)
# in_shape's last axis is K. calls_per_fold is from perf/b2z2_actsweep/out/census_whglx_c6.json.
CENSUS = [
    ("pairT_z",    [1, 16, 512, 128],  512,  "silu",    False, L1,   8960, "tenstorrent.py:7182 pair Transition chunk"),
    ("msaT",       [1, 16, 512, 64],   256,  "silu",    False, L1,   1024, "tenstorrent.py:7182 MSA Transition chunk"),
    ("ditS_o",     [1, 512, 768],      768,  "sigmoid", True,  None, 4800, "tenstorrent.py:8502 DiT output gate"),
    ("atom2tok",   [1, 4480, 128],     768,  "relu",    False, None,  200, "tenstorrent.py:9487 atom->token"),
    ("sT_768",     [1, 512, 768],      1536, "silu",    False, L1,    400, "tenstorrent.py:7182 single Transition, K=768"),
    ("sT_384",     [1, 512, 384],      1536, "silu",    False, L1,    264, "tenstorrent.py:7182 single Transition, K=384"),
    ("ditS_o_sm",  [1, 140, 32, 128],  128,  "sigmoid", True,  None,    6, "tenstorrent.py:8502 atom-level DiT gate"),
]

# The rule sweep: one output shape, K walked over the range the census spans, four activations.
# Output elems held at 16*512*512 / (N) so the SFPU pass sees the same number of tiles at every K.
KS = [32, 64, 128, 256, 384, 768, 1536]
KSWEEP = [(f"k{k}_{act}", [1, 16, 512, k], 512, act, False, L1, 0, f"rule sweep K={k}")
          for act in ("silu", "sigmoid", "relu", "gelu") for k in KS]

# The second axis. The K sweep holds output tiles fixed and moves K; this holds K fixed and moves
# the output tile count, in L1 and in DRAM, because the census's two losing sigmoid sites differ
# from the winning silu sites in BOTH (384 tiles to DRAM against 4096 tiles in L1) and one
# measurement cannot separate them.
TILES = [1, 2, 4, 8, 16]
TILESWEEP = [(f"t{r*256}_{act}_{'l1' if mc is L1 else 'dram'}", [1, r, 512, 768], 512, act,
              False, mc, 0, f"tile sweep {r*256} tiles, {'L1' if mc is L1 else 'DRAM'}")
             for act in ("silu", "sigmoid") for mc in (L1, None) for r in TILES]
# The production DiT gate shape, the only difference being where its output lands.
TILESWEEP += [("ditS_o_l1", [1, 512, 768], 768, "sigmoid", True, L1, 0,
               "tenstorrent.py:8502 shape with an L1 output instead of DRAM")]

specs = []
if a.only in ("all", "tiles"):
    specs += TILESWEEP
if a.only in ("all", "census"):
    specs += CENSUS
if a.only in ("all", "ksweep"):
    specs += KSWEEP

out = {"doc": __doc__,
       "env": {"host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
               "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
               "commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
               "reps": a.reps, "inner": a.inner, "torch": torch.__version__,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
       "specs": []}
a.out.parent.mkdir(parents=True, exist_ok=True)


def dump():
    a.out.write_text(json.dumps(out, indent=1))


def run_spec(name, in_shape, N, act, use_bias, memcfg, calls, note):
    K = in_shape[-1]
    torch.manual_seed(0)
    xt = (torch.randn(*in_shape) * 0.5).bfloat16()
    wt = (torch.randn(K, N) * (1.0 / K ** 0.5)).bfloat16()
    bt = (torch.randn(1, N) * 0.05).bfloat16() if use_bias else None
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=L1)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=DRAM)
    b = (ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=DRAM) if use_bias else None)
    ref = xt.float() @ wt.float()
    if use_bias:
        ref = ref + bt.float()
    ref = REF[act](ref)

    kw = dict(bias=b, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    if memcfg is not None:
        kw["memory_config"] = memcfg
    amc = {"memory_config": memcfg} if memcfg is not None else {}

    def fused():
        return ttnn.linear(x, w, activation=act, **kw)

    def plain():
        return ttnn.linear(x, w, **kw)

    def split():
        y = ttnn.linear(x, w, **kw)
        return ACT[act](y, output_tensor=y, **amc)

    arms = {"fused": fused, "fused2": fused, "split": split, "plain": plain}

    # accuracy, one evaluation per arm, against fp32 truth for the identical algebra
    acc = {}
    for an in ("fused", "split"):
        o = arms[an]()
        v = ttnn.to_torch(o).float().reshape(ref.shape)
        ttnn.deallocate(o)
        d = (v - ref).abs()
        acc[an] = {"rms_vs_fp32": float((d ** 2).mean().sqrt()),
                   "max_abs_vs_fp32": float(d.max()),
                   "mean_rel_vs_fp32": float(d.sum() / ref.abs().sum().clamp_min(1e-9))}

    for _ in range(2):
        for fn in arms.values():
            ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)

    samples = {k: [] for k in arms}
    for _ in range(a.reps):
        for an, fn in arms.items():                       # interleaved: one rep = one of each arm
            t0 = time.perf_counter()
            for _ in range(a.inner):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            samples[an].append((time.perf_counter() - t0) / a.inner * 1e3)

    med = {k: st.median(v) for k, v in samples.items()}
    ttnn.deallocate(x); ttnn.deallocate(w)
    if b is not None:
        ttnn.deallocate(b)
    out_elems = 1
    for d_ in in_shape[:-1]:
        out_elems *= d_
    out_elems *= N
    r = {"name": name, "note": note, "in_shape": in_shape, "K": K, "N": N, "activation": act,
         "bias": use_bias, "calls_per_fold": calls,
         "out_elems": out_elems, "out_tiles": out_elems // 1024,
         "ms": {k: round(v, 6) for k, v in med.items()},
         "spread_pct": {k: round((max(v) - min(v)) / st.median(v) * 100, 2)
                        for k, v in samples.items()},
         "aa_floor": round(med["fused"] / med["fused2"], 5),
         "ratio_split": round(med["fused"] / med["split"], 5),
         "fused_penalty_ms": round(med["fused"] - med["plain"], 6),
         "split_penalty_ms": round(med["split"] - med["plain"], 6),
         "accuracy": acc}
    r["saving_ms_per_fold"] = round((med["fused"] - med["split"]) * calls, 3) if calls else None
    out["specs"].append(r)
    dump()
    print(f"{name:12s} K={K:<5d} N={N:<5d} {act:8s} tiles={r['out_tiles']:<6d} "
          f"fused={med['fused']:.5f} split={med['split']:.5f} plain={med['plain']:.5f} "
          f"ratio={r['ratio_split']:.4f} aa={r['aa_floor']:.4f}", flush=True)
    return r


for s in specs:
    try:
        run_spec(*s)
    except Exception as e:                                      # a shape that will not build is data
        out["specs"].append({"name": s[0], "error": f"{type(e).__name__}: {e}"[:400],
                             "in_shape": s[1], "N": s[2], "activation": s[3]})
        dump()
        print(f"{s[0]:12s} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
dump()
