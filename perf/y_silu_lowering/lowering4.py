#!/usr/bin/env python3
"""y-silu-lowering round 4 -- the two claims the doc left unmeasured.

Round 3 settled WHY the fused silu costs 2x a standalone one (DST_ACCUM_MODE picks an accurate
sigmoid, not a defect). It left two of its own statements as reasoning rather than measurement:

  A. the shape dependence. The fused silu costs 0.0179 us per padded output tile at the 4-D pair
     shape and 0.0037 at the 3-D c_s=384 site. Round 3 offered launch-floor domination as the
     mechanism and named the experiment that kills it: sweep the work per core and watch the
     per-tile penalty climb to the asymptote. If it stays flat, the mechanism is wrong.
     Two sweeps here, because dimensionality and work-per-core are confounded in the two production
     shapes: batch at the 4-D shape, and M at the 3-D shape. If a small-batch 4-D shape is as cheap
     per tile as the 3-D one, dimensionality is refuted outright.

  B. the 512 aa figure, which round 3 extrapolated as (512/298)^2. Read from the source rather than
     assumed: `TRANSITION_H_CHUNK_SIZE_BIG = 32` is gated on `W <= 384` (tenstorrent.py:2411), so at
     W=512 the row chunk is 16, not 32, and a Transition call runs 32 fc1 silus rather than 10 at a
     different shape. Measure that shape.

Every arm is the production compute kernel config (HiFi4, fp32_dest_acc_en, packer_l1_acc) with its
own bare-matmul control re-measured in the same pass, arms alternated within a shape.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn


def med(v):
    return sorted(v)[len(v) // 2]


def timed(dev, fn, k, reps=3, warm=3):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(k):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t) / k * 1e6)
    return round(med(out), 3)


def tiles(shape, n):
    """Padded output tiles for in0 `shape` against a [K, n] weight."""
    m = -(-shape[-2] // 32) * (-(-n // 32))
    for d in shape[:-2]:
        m *= d
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "lowering4.json"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    import tt_bio.tenstorrent as T
    dev = T.get_device()
    L1, CG = ttnn.L1_MEMORY_CONFIG, T.CORE_GRID_MAIN
    ckc = {"on": ttnn.init_device_compute_kernel_config(
               dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
               packer_l1_acc=True),
           "off": ttnn.init_device_compute_kernel_config(
               dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False,
               packer_l1_acc=True)}
    torch.manual_seed(0)

    # (label, in0 shape, K, N, which dest-accum settings to run)
    CASES = [
        # -- the 298 aa production fc1 shape, this session's anchor against rounds 1-3 --------------
        ("pair298_b30", (1, 30, 298, 256), 256, 1024, ("on", "off")),
        # -- the REAL 512 aa fc1 shape: row chunk 16, not 32 (W=512 misses the W<=384 gate) --------
        ("pair512_b16", (1, 16, 512, 256), 256, 1024, ("on", "off")),
        # a 32-row chunk at 512 too, so the chunk-size change is separated from the size change
        ("pair512_b32", (1, 32, 512, 256), 256, 1024, ("on",)),
        # -- sweep A1: batch at the 4-D pair shape. work per core is the variable, 4-D is fixed ----
        ("pair298_b1",  (1, 1, 298, 256), 256, 1024, ("on",)),
        ("pair298_b2",  (1, 2, 298, 256), 256, 1024, ("on",)),
        ("pair298_b4",  (1, 4, 298, 256), 256, 1024, ("on",)),
        ("pair298_b8",  (1, 8, 298, 256), 256, 1024, ("on",)),
        ("pair298_b16", (1, 16, 298, 256), 256, 1024, ("on",)),
        # -- sweep A2: M at the 3-D transition_s shape. the experiment round 3 named and skipped ---
        ("cs384_m298",  (1, 298, 384), 384, 1536, ("on",)),
        ("cs384_m596",  (1, 596, 384), 384, 1536, ("on",)),
        ("cs384_m1192", (1, 1192, 384), 384, 1536, ("on",)),
        ("cs384_m2384", (1, 2384, 384), 384, 1536, ("on",)),
        ("cs384_m4768", (1, 4768, 384), 384, 1536, ("on",)),
        ("cs384_m9536", (1, 9536, 384), 384, 1536, ("on",)),
    ]

    res = {"arch": str(dev.arch()), "grid": str(CG), "k": a.k, "passes": a.passes,
           "load_start": [round(v, 2) for v in os.getloadavg()], "cases": {}}
    for label, shp, K, N, accs in CASES:
      try:
        xt = torch.randn(*shp, dtype=torch.bfloat16)
        wt = (torch.randn(K, N) * 0.05).to(torch.bfloat16)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=L1)
        w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=L1)
        arms = [(acc, act) for acc in accs for act in (None, "silu")]
        raw = {f"{acc}_{act}": [] for acc, act in arms}
        for p in range(a.passes):
            for acc, act in arms:
                def fn(act=act, acc=acc):
                    y = ttnn.linear(x, w, activation=act, compute_kernel_config=ckc[acc],
                                    memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG)
                    ttnn.deallocate(y)
                raw[f"{acc}_{act}"].append(timed(dev, fn, a.k))
        m = {k: med(v) for k, v in raw.items()}
        nt = tiles(shp, N)
        row = {"shape": list(shp), "K": K, "N": N, "out_tiles": nt, "raw": raw, "median": m,
               "load": [round(v, 2) for v in os.getloadavg()]}
        for acc in accs:
            pen = m[f"{acc}_silu"] - m[f"{acc}_None"]
            row[f"penalty_{acc}"] = round(pen, 3)
            row[f"penalty_{acc}_per_tile"] = round(pen / nt, 6)
        if "off" in accs:
            row["gap"] = round(row["penalty_on"] - row["penalty_off"], 3)
            row["gap_per_tile"] = round(row["gap"] / nt, 6)
        res["cases"][label] = row
        print(label, "tiles", nt, json.dumps(m), json.dumps({k: v for k, v in row.items()
                                              if k.startswith(("penalty", "gap"))}),
              "load", row["load"], flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
      except Exception as e:
        res["cases"][label] = {"shape": list(shp), "error": f"{type(e).__name__}: {e}"[:400]}
        print(label, "FAILED", type(e).__name__, str(e)[:200], flush=True)
      res["load_end"] = [round(v, 2) for v in os.getloadavg()]
      Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out, flush=True)


main()
