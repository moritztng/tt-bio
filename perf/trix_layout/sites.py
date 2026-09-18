#!/usr/bin/env python3
"""Site census for the trimul's LAYOUT class: every op that moves bytes so the next op can
read them, with its cost, on today's shipped defaults.

`perf/trimul_abs/tape2.py` tapes the same module but predates the gated move, the row-blocked
in-projection and the deferred matmul transpose, and its fixture module (`pf_layer`) is gone.
This builds the layer the way `perf/b2z2_byte_round2/bitexact_gout.py` does -- real
`reference.PairformerLayer` weights with the zeroed output projections given values, so every
sub-unit reaches the output -- and wraps the layout entry points the CURRENT default path takes.

Wall is pipelined (no per-call sync); the taped pass syncs around every op, so the taped sum
over-reads. The share is what this is for, not the absolute.
"""
import collections
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# Score THIS checkout, not whatever tt_bio the env has installed: the shared
# /home/ttuser/tt-bio-dev is first on sys.path otherwise.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from clocksample import during  # noqa: E402

import tt_bio  # noqa: E402
assert str(REPO) in tt_bio.__file__, f"wrong tt_bio: {tt_bio.__file__}"
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import reblock_permute as RB  # noqa: E402
from tt_bio import mm_generic as MG  # noqa: E402
from tt_bio import trimul_tail as TL  # noqa: E402

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
TAG = sys.argv[2] if len(sys.argv) > 2 else "qb1c0"
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
weights = {k: v.float() for k, v in rl.state_dict().items()}
for k, v in list(weights.items()):
    if v.numel() and float(v.abs().max()) == 0.0:
        weights[k] = torch.randn_like(v) * 0.05
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)

f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
z = f(torch.randn(1, S, S, 128))
m1 = torch.ones(1, S)
mask_tt = f(m1[:, :, None] * m1[:, None, :])

ROWS, ON = [], [False]

# Which class each taped entry point belongs to. LAYOUT is the class this row owns: ops that
# move bytes and compute nothing. FUSED-LAYOUT moves bytes AND does arithmetic on the way.
CLASS = {
    "permute": "LAYOUT", "transpose": "LAYOUT", "reallocate": "LAYOUT", "concat": "LAYOUT",
    "clone": "LAYOUT", "slice": "LAYOUT", "to_memory_config": "LAYOUT", "reshape": "LAYOUT",
    "unsqueeze": "LAYOUT", "chunk": "LAYOUT", "typecast": "LAYOUT",
    "REBLOCK_fwd": "LAYOUT", "REBLOCK_back": "LAYOUT",
    "REBLOCK_gated": "FUSED-LAYOUT",
    "generic_minimal_matmul": "COMPUTE", "fused_tail": "COMPUTE",
    "matmul": "COMPUTE", "linear": "COMPUTE", "minimal_matmul": "COMPUTE",
    "layer_norm": "COMPUTE", "multiply_": "COMPUTE", "fused_tail": "COMPUTE",
}


def shp(t):
    try:
        return "x".join(str(int(d)) for d in t.shape)
    except Exception:
        return "?"


def buf(kw):
    mc = kw.get("memory_config")
    if mc is None:
        return "-"
    return "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"


def wrap(mod, name, key, tagger):
    orig = getattr(mod, name)

    def g(*a, **kw):
        if not ON[0]:
            return orig(*a, **kw)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        ttnn.synchronize_device(dev)
        ROWS.append((key, tagger(a, kw), (time.perf_counter() - t0) * 1e3))
        return out
    setattr(mod, name, g)


wrap(ttnn, "matmul", "matmul", lambda a, kw: f"[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn, "linear", "linear", lambda a, kw: f"[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn.experimental, "minimal_matmul", "minimal_matmul",
     lambda a, kw: f"[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn, "permute", "permute", lambda a, kw: f"{tuple(a[1])}[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "transpose", "transpose", lambda a, kw: f"({a[1]},{a[2]})[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "layer_norm", "layer_norm", lambda a, kw: f"[{shp(a[0])}]")
wrap(ttnn, "multiply_", "multiply_", lambda a, kw: f"[{shp(a[0])}]")
wrap(ttnn, "chunk", "chunk", lambda a, kw: f"{kw.get('chunks','')}[{shp(a[0])}]")
wrap(ttnn, "concat", "concat",
     lambda a, kw: f"[{len(a[0])}x{shp(a[0][0])}]dim"
     + str(a[1] if len(a) > 1 else kw.get("dim")))
wrap(ttnn, "clone", "clone", lambda a, kw: f"[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "reallocate", "reallocate", lambda a, kw: f"[{shp(a[0])}]")
wrap(ttnn, "slice", "slice", lambda a, kw: f"[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "to_memory_config", "to_memory_config", lambda a, kw: f"[{shp(a[0])}]")
wrap(ttnn, "reshape", "reshape", lambda a, kw: f"[{shp(a[0])}]")
wrap(ttnn, "unsqueeze", "unsqueeze", lambda a, kw: f"[{shp(a[0])}]dim{a[1]}")
wrap(ttnn, "typecast", "typecast", lambda a, kw: f"[{shp(a[0])}]")
wrap(RB, "reblock_permute", "REBLOCK_fwd", lambda a, kw: f"(0,3,1,2)[{shp(a[0])}]->{buf(kw)}")
wrap(RB, "reblock_permute_back", "REBLOCK_back", lambda a, kw: f"(0,2,3,1)[{shp(a[0])}]->{buf(kw)}")
wrap(RB, "reblock_permute_gated", "REBLOCK_gated",
     lambda a, kw: f"[{shp(a[0])}]c{a[3] if len(a) > 3 else '?'}->{buf(kw)}")

wrap(MG, "generic_minimal_matmul", "generic_minimal_matmul",
     lambda a, kw: f"[{shp(a[0])}@{shp(a[1])}]")
wrap(TL, "fused_tail", "fused_tail", lambda a, kw: f"[{shp(a[0])}]")

MODS = {"start": layer.triangle_multiplication_start,
        "end": layer.triangle_multiplication_end}


def census(tm, label):
    global ROWS
    call = lambda: tm(z, mask_tt)
    for _ in range(3):
        ttnn.deallocate(call())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(6):
        ttnn.deallocate(call())
    ttnn.synchronize_device(dev)
    wall = (time.perf_counter() - t0) * 1e3 / 6
    ROWS = []
    ON[0] = True
    ttnn.deallocate(call())
    ON[0] = False
    agg = collections.OrderedDict()
    for key, tag, ms in ROWS:
        a = agg.setdefault((key, tag), dict(key=key, cls=CLASS.get(key, "?"), tag=tag, n=0, ms=0.0))
        a["n"] += 1
        a["ms"] += ms
    rows = sorted(agg.values(), key=lambda a: -a["ms"])
    total = sum(a["ms"] for a in rows)
    by_cls = collections.Counter()
    for a in rows:
        by_cls[a["cls"]] += a["ms"]
    print(f"\n=== trimul({label}) N={S}: pipelined wall {wall:.3f} ms | taped sum {total:.3f} ms "
          f"({total/wall:.3f}x, syncs included) ===")
    for c, v in by_cls.most_common():
        print(f"    {c:14s} {v:8.3f} ms  {100*v/total:5.1f}% of taped")
    print()
    for a in rows:
        a["ms"] = round(a["ms"], 4)
        a["share"] = round(a["ms"] / total, 4)
        print(f"  {a['cls']:13s} {a['key']:17s} {a['tag']:46s} {a['n']:3d} {a['ms']:8.3f} "
              f"{100*a['share']:5.1f}%")
    return dict(label=label, wall_ms=round(wall, 4), taped_sum_ms=round(total, 4),
                by_class={k: round(v, 4) for k, v in by_cls.items()}, ops=rows)


OUT = {"n": S, "host": TAG, "grid": list(tt.COMPUTE_GRID_MAIN)}
with during(period=1.5) as clk:
    OUT["census"] = [census(MODS[k], k) for k in ("start", "end")]
OUT["clock"] = clk.summary()
OUT["branch_stats"] = {str(k): v for k, v in tt.TRIMUL_MM_TRANSPOSE_STATS.items()}
OUT["reblock_stats"] = {"served": int(RB.STATS[0]), "declined": int(RB.STATS[1]),
                        "gated": int(RB.STATS_GATED[0])}
print("\n" + clk.line(0))
print("branch census:", OUT["branch_stats"])
print("reblock:", OUT["reblock_stats"])
p = HERE / f"sites_{S}_{TAG}.json"
p.write_text(json.dumps(OUT, indent=2))
print("WROTE", p)
