#!/usr/bin/env python3
"""X3 closeout probes on qb1 card 2: the slice-alignment cost, the untilize cliff's mechanism,
the transition fixed term, and the call counts that C4 and C5 are priced against (counted in a
live fold, not derived)."""
from __future__ import annotations
import importlib.util, json, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.protenix as P
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
log = lambda *a: print(*a, file=sys.stderr, flush=True)
res = {}


def timed(fn, warm=2, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o) * 1e6


# ---- A. the slice start-offset penalty on this card (P5 measured 7.49 vs 202.42 us on qb2) ----
log("=== A. ttnn slice start offset, 4096x4096 source, identical output bytes ===")
src = ttnn.from_torch(torch.zeros(4096, 4096, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                      device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
sl = {}
for tag, f in (("[0:256, 0:32] aligned", lambda: src[0:256, 0:32]),
               ("[0:256, 32:64] aligned", lambda: src[0:256, 32:64]),
               ("[0:256, 1:33] UNALIGNED", lambda: src[0:256, 1:33]),
               ("[0:256, 0:1] aligned sub-tile", lambda: src[0:256, 0:1]),
               ("[0:256, 1:2] UNALIGNED sub-tile", lambda: src[0:256, 1:2]),
               ("[0:256, 8:16] UNALIGNED (PWA m/g head 1)", lambda: src[0:256, 8:16])):
    us = timed(lambda: ttnn.deallocate(f()))
    sl[tag] = round(us, 2)
    log(f"  {tag:44s} {us:9.2f} us")
ttnn.deallocate(src)
res["slice_alignment"] = sl

# ---- B. is the untilize cliff a single-core fallback? ------------------------------------
log("=== B. untilize cliff mechanism: multicore vs forced single core ===")
mech = {}
for rows_t, cols_t, tag in ((298, 298, "298x298 tiles GOOD"), (256, 256, "256x256 tiles BAD")):
    t = ttnn.from_torch(torch.zeros(rows_t * 32, cols_t * 32, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    nb = rows_t * cols_t * 2048
    row = {"MB": round(nb / 2**20, 1)}
    for lbl, fn in (("to_layout", lambda t=t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)),
                    ("untilize mc=True", lambda t=t: ttnn.untilize(t, use_multicore=True)),
                    ("untilize mc=False", lambda t=t: ttnn.untilize(t, use_multicore=False))):
        try:
            us = timed(lambda: ttnn.deallocate(fn()), warm=1, pipe=1, reps=3)
            row[lbl] = {"us": round(us, 1), "GB/s": round(2 * nb / (us * 1e-6) / 1e9, 1)}
            log(f"  {tag:20s} {lbl:20s} {us:10.1f} us  {2*nb/(us*1e-6)/1e9:7.1f} GB/s")
        except Exception as e:
            row[lbl] = f"ERR {str(e)[:80]}"
            log(f"  {tag:20s} {lbl:20s} ERR {str(e)[:80]}")
    mech[tag] = row
    ttnn.deallocate(t)
res["cliff_mechanism"] = mech

# ---- C. C4: the transition fixed term at the MSA transition's own shape, on this card ------
log("=== C. C4 row sweep at the MSA transition shape, L1 resident ===")
c4 = {}
w = ttnn.from_torch(torch.zeros(128, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
ln_w = ttnn.from_torch(torch.ones(128, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                       device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
for mult in (1, 2, 4):
    try:
        x = ttnn.from_torch(torch.zeros(35 * mult, 320, 128, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
    except Exception:
        x = ttnn.from_torch(torch.zeros(35 * mult, 320, 128, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    r = {}
    r["layer_norm"] = round(timed(lambda: ttnn.deallocate(
        ttnn.layer_norm(x, weight=ln_w, epsilon=1e-5, compute_kernel_config=ckc))), 2)
    r["linear"] = round(timed(lambda: ttnn.deallocate(
        ttnn.linear(x, w, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN))), 2)
    r["linear_silu"] = round(timed(lambda: ttnn.deallocate(
        ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN))), 2)
    c4[mult] = r
    log(f"  mult={mult}  layer_norm {r['layer_norm']:8.2f}  linear {r['linear']:8.2f}  "
        f"linear+silu {r['linear_silu']:8.2f} us")
    ttnn.deallocate(x)
res["c4_rows"] = c4

# ---- D. call counts, counted in a live fold -----------------------------------------------
log("=== D. call counts in a live 298 aa fold ===")
COUNT = {}


def count(name, fn):
    def w(*a, **k):
        COUNT[name] = COUNT.get(name, 0) + 1
        return fn(*a, **k)
    return w


T.Transition.__call__ = count("Transition.__call__", T.Transition.__call__)
T.PairWeightedAveraging.__call__ = count("PWA.__call__", T.PairWeightedAveraging.__call__)
_lin = P.Trunk._lin
P.Trunk._lin = count("Trunk._lin", _lin)
_tpl = P.Trunk._template
P.Trunk._template = count("Trunk._template", _tpl)

spec = importlib.util.spec_from_file_location(
    "tt_baseline", REPO / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
one_fold, meta, _ = tb.build_fold("protenix-v2", Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser(),
                                  REPO / "examples/prot300.yaml",
                                  REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
one_fold()
COUNT.clear()
wall, _ = one_fold()
res["fold_counts"] = dict(COUNT)
res["fold_s"] = round(wall, 3)
log(f"  fold {wall:.3f}s  " + "  ".join(f"{k}={v}" for k, v in sorted(COUNT.items())))

print(json.dumps(res, indent=1, default=str))
