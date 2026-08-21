"""Per-call cost of every softmax arm at the shapes Boltz-2's own census saw, then per-fold cost.

Prices the two guards the masked-arm scoring left tied on accuracy. Md floors `d` before exp, one
extra op on the FULL score tensor, which at Fp32TriangleAttention is the largest softmax tensor in
the repo. Ms floors the row sum, one op on the [..,1] reduction. If Md is not measurably worse than
Ms, the guard choice is free and Md wins because it is the only one finite on a fully-masked row.

Isolated per-op timing oversyncs and inflates marginal cost (tt-bio-isolated-op-timing-oversync-
inflates-cost measured ~2x), so treat the ratios here as an upper bound on what a fold pays and let
the full-fold A/B be the authority.
"""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, "/home/ttuser/.coworker/wt/accurate-softmax-crossmodel")
import torch, ttnn
from tt_bio.tenstorrent import _accurate_softmax

ROOT = "/home/ttuser/.coworker/wt/accurate-softmax-crossmodel"
WARM, REP = 3, 10

# (shape, dtype, n_calls, site, path) straight out of the two Boltz-2 census JSONs.
CELLS = [
    ((192, 4, 192, 192), ttnn.float32,  768, "4644 Fp32TriangleAttention", "affinity"),
    ((1, 16, 192, 192),  ttnn.float32,  384, "4787 Fp32AttentionPairBias", "affinity"),
    ((1, 16, 192, 192),  ttnn.bfloat16, 560, "4058 AttentionPairBias",     "affinity"),
    ((1, 192, 192),      ttnn.bfloat16, 320, "5479 PairWeightedAveraging", "affinity"),
    ((1, 16, 256, 256),  ttnn.bfloat16, 408, "4058 AttentionPairBias",     "structure"),
    ((1, 256, 256),      ttnn.bfloat16, 128, "5479 PairWeightedAveraging", "structure"),
]


def chain(x, floor_d=None, floor_s=None):
    m = ttnn.max(x, dim=-1, keepdim=True)
    d = ttnn.subtract(x, m)
    ttnn.deallocate(m)
    if floor_d is not None:
        d = ttnn.maximum(d, floor_d)
    ttnn.exp(d, output_tensor=d)
    s = ttnn.sum(d, dim=-1, keepdim=True)
    if floor_s is not None:
        s = ttnn.maximum(s, floor_s)
    p = ttnn.divide(d, s)
    ttnn.deallocate(d); ttnn.deallocate(s)
    return p


ARMS = {
    "F":  lambda x: ttnn.softmax(x, dim=-1),
    "M":  lambda x: _accurate_softmax(x),
    "Md": lambda x: chain(x, floor_d=-80.0),
    "Ms": lambda x: chain(x, floor_s=1e-30),
}


def bench(dev, shape, dt):
    torch.manual_seed(0)
    host = torch.randn(*shape, dtype=torch.float32) * 3.0
    x = ttnn.from_torch(host if dt == ttnn.float32 else host.to(torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    out = {}
    for name, fn in ARMS.items():
        for _ in range(WARM):
            ttnn.deallocate(fn(x))
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(REP):
            t0 = time.perf_counter()
            r = fn(x)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(r)
        out[name] = {"median_ms": statistics.median(ts),
                     "min_ms": min(ts), "max_ms": max(ts)}
    ttnn.deallocate(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "perf/xmsoftmax/results/arm_cost_boltz2.json"))
    ap.add_argument("--fold-ms-structure", type=float, required=True)
    ap.add_argument("--fold-ms-affinity", type=float, required=True)
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0)
    rows = []
    try:
        for shape, dt, n, site, path in CELLS:
            c = bench(dev, list(shape), dt)
            rows.append({"shape": list(shape), "dtype": str(dt), "n_calls": n,
                         "site": site, "path": path, "arms": c})
            print("%-20s %-9s %-30s n=%-5d F=%7.3f  M=%7.3f (%.2fx)  Md=%7.3f (%.2fx)  Ms=%7.3f (%.2fx)"
                  % (str(list(shape)), str(dt).replace("DataType.", ""), site, n,
                     c["F"]["median_ms"],
                     c["M"]["median_ms"],  c["M"]["median_ms"] / c["F"]["median_ms"],
                     c["Md"]["median_ms"], c["Md"]["median_ms"] / c["F"]["median_ms"],
                     c["Ms"]["median_ms"], c["Ms"]["median_ms"] / c["F"]["median_ms"]))
    finally:
        ttnn.close_device(dev)

    fold = {"structure": a.fold_ms_structure, "affinity": a.fold_ms_affinity}
    res = {"cells": rows, "fold_ms": fold, "paths": {}}
    print("\n=== predicted per-fold softmax cost ===")
    for path in ("structure", "affinity"):
        sel = [r for r in rows if r["path"] == path]
        tot = {k: sum(r["arms"][k]["median_ms"] * r["n_calls"] for r in sel)
               for k in ("F", "M", "Md", "Ms")}
        # the fp32-only flip: sites 4644 + 4787, the ones this task would add a lever to
        fp32 = [r for r in sel if "Fp32" in r["site"]]
        fp32tot = {k: sum(r["arms"][k]["median_ms"] * r["n_calls"] for r in fp32)
                   for k in ("F", "M", "Md", "Ms")}
        d = {"softmax_ms": tot, "fp32_sites_ms": fp32tot,
             "softmax_share_of_fold": tot["F"] / fold[path]}
        for k in ("M", "Md", "Ms"):
            d["pred_fold_slowdown_all_" + k] = (tot[k] - tot["F"]) / fold[path]
            d["pred_fold_slowdown_fp32only_" + k] = (fp32tot[k] - fp32tot["F"]) / fold[path]
        res["paths"][path] = d
        print("%-10s fold %7.0f ms | softmax F %8.1f ms (%5.1f%% of fold)" %
              (path, fold[path], tot["F"], 100 * d["softmax_share_of_fold"]))
        for k in ("M", "Md", "Ms"):
            print("             %-3s all sites -> %+6.1f%% fold   |  fp32 sites only -> %+6.1f%% fold"
                  % (k, 100 * d["pred_fold_slowdown_all_" + k],
                     100 * d["pred_fold_slowdown_fp32only_" + k]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
