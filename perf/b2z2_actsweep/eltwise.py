#!/usr/bin/env python3
"""The other half of the census: an SFPU pass fused into a BINARY ELTWISE op, not into a matmul.

`census.py` finds 31,512 such calls in a 512 aa fold moving 110.5 GB of output -- more than the
85.7 GB the matmul-fused sites move. They are the sigmoid gates (`x * sigmoid(g)`) and the diffusion
SwiGLU (`silu(a) * g`), written as `input_tensor_{a,b}_activations=[...]`.

The matmul result says the fused SFPU pass runs at roughly half the rate the standalone op reaches.
If that is a property of the fused pass and not of the matmul kernel, it should show here too --
except the arithmetic is against it: splitting a matmul's activation adds an L1 round trip, while
splitting an eltwise activation adds an ENTIRE SECOND OP over the same operand. So this is the
falsifier for "the property is about the fused pass": if the penalty is a fused-pass property it
survives the op it is fused into; if it is a matmul-kernel property it vanishes here.

  fused    ttnn.multiply(x, g, input_tensor_b_activations=[SIGMOID])
  split    ttnn.sigmoid(g) then ttnn.multiply(x, g_act)
  plain    ttnn.multiply(x, g)                                        decomposition
  act      ttnn.sigmoid(g) alone                                      what the pass costs on its own

    eltwise.py --out <json> [--reps 9] [--inner 32]
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
a = ap.parse_args()

import torch                                                                   # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                    # noqa: E402
import tt_bio.tenstorrent as T                                                 # noqa: E402
import tt_bio as _TB                                                           # noqa: E402
assert Path(_TB.__file__).resolve().is_relative_to(ROOT), _TB.__file__

dev = T.get_device()
g_ = dev.compute_with_storage_grid_size()
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
UOP = {"sigmoid": ttnn.UnaryOpType.SIGMOID, "silu": ttnn.UnaryOpType.SILU}
FN = {"sigmoid": ttnn.sigmoid, "silu": ttnn.silu}
TREF = {"sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu}

# (name, shape, act, operand, calls_per_fold, note) -- shapes and counts from census_whglx_c6.json
SPECS = [
    ("trimul_tail",  [1, 512, 512, 128],  "sigmoid", "b",  560, "tenstorrent.py:5958 trimul out gate"),
    ("triatt_gate",  [512, 4, 512, 32],   "sigmoid", "b",  560, "tenstorrent.py:6478 tri-attention gate"),
    ("adaln",        [1, 512, 768],       "sigmoid", "b", 9600, "tenstorrent.py:8351 AdaLN scale"),
    ("ctb_swiglu",   [1, 512, 1536],      "silu",    "a", 4800, "tenstorrent.py:8404 ConditionedTransitionBlock"),
    ("pwa_head",     [1024, 512, 32],     "sigmoid", "b",  128, "tenstorrent.py:8678 PairWeightedAveraging"),
    ("apb_gate",     [1, 512, 768],       "sigmoid", "b", 4800, "tenstorrent.py:7145 AttentionPairBias"),
    ("ctb_out",      [1, 512, 768],       "sigmoid", "a", 4800, "tenstorrent.py:8432 CTB output gate"),
    ("adaln_atom",   [1, 140, 32, 128],   "sigmoid", "b", 2400, "tenstorrent.py:8351 atom-level AdaLN"),
    ("ctb_sw_atom",  [1, 140, 32, 256],   "silu",    "a", 1200, "tenstorrent.py:8404 atom-level SwiGLU"),
    ("apb_small",    [1, 512, 384],       "sigmoid", "b",  264, "tenstorrent.py:7145 trunk s gate"),
]

out = {"doc": __doc__,
       "env": {"host": socket.gethostname(), "grid": [g_.x, g_.y], "arch": str(dev.arch()),
               "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
               "commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
               "reps": a.reps, "inner": a.inner,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
       "specs": []}
a.out.parent.mkdir(parents=True, exist_ok=True)


def dump():
    a.out.write_text(json.dumps(out, indent=1))


def run(name, shape, act, operand, calls, note):
    torch.manual_seed(0)
    xt = (torch.randn(*shape) * 0.5).bfloat16()
    gt = (torch.randn(*shape) * 0.5).bfloat16()
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=L1)
    x, gg = mk(xt), mk(gt)
    kwa = ("input_tensor_a_activations" if operand == "a" else "input_tensor_b_activations")
    ref = (TREF[act](xt.float()) * gt.float() if operand == "a"
           else xt.float() * TREF[act](gt.float()))

    def fused():
        return ttnn.multiply(x, gg, **{kwa: [UOP[act]]}, memory_config=L1)

    def split():
        t = FN[act](x if operand == "a" else gg, memory_config=L1)
        r = ttnn.multiply(t, gg, memory_config=L1) if operand == "a" \
            else ttnn.multiply(x, t, memory_config=L1)
        ttnn.deallocate(t)
        return r

    def plain():
        return ttnn.multiply(x, gg, memory_config=L1)

    def actonly():
        return FN[act](gg, memory_config=L1)

    arms = {"fused": fused, "fused2": fused, "split": split, "plain": plain, "act": actonly}

    acc = {}
    for an in ("fused", "split"):
        o = arms[an]()
        v = ttnn.to_torch(o).float().reshape(ref.shape)
        ttnn.deallocate(o)
        d = (v - ref).abs()
        acc[an] = {"rms_vs_fp32": float((d ** 2).mean().sqrt()),
                   "max_abs_vs_fp32": float(d.max())}

    for _ in range(2):
        for fn in arms.values():
            ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    s = {k: [] for k in arms}
    for _ in range(a.reps):
        for an, fn in arms.items():
            t0 = time.perf_counter()
            for _ in range(a.inner):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            s[an].append((time.perf_counter() - t0) / a.inner * 1e3)
    med = {k: st.median(v) for k, v in s.items()}
    ttnn.deallocate(x); ttnn.deallocate(gg)
    elems = 1
    for d_ in shape:
        elems *= d_
    r = {"name": name, "note": note, "shape": shape, "activation": act, "operand": operand,
         "calls_per_fold": calls, "out_tiles": elems // 1024,
         "ms": {k: round(v, 6) for k, v in med.items()},
         "aa_floor": round(med["fused"] / med["fused2"], 5),
         "ratio_split": round(med["fused"] / med["split"], 5),
         "fused_penalty_ms": round(med["fused"] - med["plain"], 6),
         "split_penalty_ms": round(med["split"] - med["plain"], 6),
         "standalone_act_ms": round(med["act"], 6),
         "saving_ms_per_fold": round((med["fused"] - med["split"]) * calls, 3),
         "accuracy": acc}
    out["specs"].append(r); dump()
    print(f"{name:12s} {act:8s} op-{operand} tiles={r['out_tiles']:<6d} fused={med['fused']:.5f} "
          f"split={med['split']:.5f} plain={med['plain']:.5f} act={med['act']:.5f} "
          f"ratio={r['ratio_split']:.4f} aa={r['aa_floor']:.4f} "
          f"fold={r['saving_ms_per_fold']:+.1f} ms", flush=True)


for sp in SPECS:
    try:
        run(*sp)
    except Exception as e:
        out["specs"].append({"name": sp[0], "error": f"{type(e).__name__}: {e}"[:400]})
        dump()
        print(f"{sp[0]:12s} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
dump()
tot = sum(r.get("saving_ms_per_fold", 0) for r in out["specs"] if "saving_ms_per_fold" in r)
print(f"TOTAL eltwise saving if every site split: {tot:+.1f} ms/fold")
