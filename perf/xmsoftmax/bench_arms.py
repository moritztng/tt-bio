"""Per-call cost of each softmax arm at the shapes the census actually saw, then per-fold cost.

The point is to predict the perf half arithmetically before spending hours on parity anchors and
size ladders: cost_per_call(arm, shape, dtype) x n_calls from the census, against the published
256 aa fold time. A prediction written down before the measurement is what makes the measurement
falsifiable (perf-method-floor-screen-predict-then-build).

Note the arms are not charged equally by dtype. `_accurate_softmax` upcasts to fp32, so at a bf16
site it pays a typecast plus fp32 intermediates that an fp32 site does not -- the 4.22x headline
from RF3 was measured on fp32 input and is a floor, not a ceiling, for the bf16 sites here.
"""
import glob
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio.tenstorrent import _accurate_softmax  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = {"DataType.FLOAT32": ttnn.float32, "DataType.BFLOAT16": ttnn.bfloat16}
FOLD_S = {"esmfold2": 19.6, "protenix-v2": 15.8, "openfold3": 11.3, "opendde": 25.2}
WARM, REP = 3, 10


def arm_F(x):
    return ttnn.softmax(x, dim=-1)


def arm_R(x):
    f = ttnn.softmax(x, dim=-1)
    return ttnn.divide(f, ttnn.sum(f, dim=-1, keepdim=True))


def arm_M(x):
    return _accurate_softmax(x)


ARMS = {"F": arm_F, "R": arm_R, "M": arm_M}


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
        out[name] = statistics.median(ts)
    ttnn.deallocate(x)
    return out


def main():
    # every (shape, dtype) the census saw, above 1% of any model's volume
    combos, models = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, "perf/xmsoftmax/results/census_*_256.json"))):
        model = os.path.basename(f)[len("census_"):-len("_256.json")]
        sites = [s for s in json.load(open(f))["sites"] if s["deficit"] is not None]
        models[model] = sites
        for s in sites:
            combos[(tuple(s["shape"]), s["dtype"])] = True

    dev = ttnn.open_device(device_id=0)
    cost = {}
    try:
        for (shape, dtype) in sorted(combos, key=lambda k: math.prod(k[0])):
            try:
                cost[(shape, dtype)] = bench(dev, list(shape), DT[dtype])
                c = cost[(shape, dtype)]
                print("%-24s %-9s F=%7.3f ms  R=%7.3f (%.2fx)  M=%7.3f (%.2fx)"
                      % (str(list(shape)), dtype.replace("DataType.", ""),
                         c["F"], c["R"], c["R"] / c["F"], c["M"], c["M"] / c["F"]))
            except Exception as e:
                print("%-24s %-9s SKIP %s" % (str(list(shape)), dtype, type(e).__name__))
    finally:
        ttnn.close_device(dev)

    print("\n=== predicted per-fold softmax cost at 256 aa ===")
    res = {"per_call_ms": {f"{list(k[0])}|{k[1]}": v for k, v in cost.items()}, "models": {}}
    for model, sites in sorted(models.items()):
        tot = {"F": 0.0, "R": 0.0, "M": 0.0}
        lever = {"F": 0.0, "M": 0.0}   # only the sites tenstorrent.py:4058 reaches
        for s in sites:
            c = cost.get((tuple(s["shape"]), s["dtype"]))
            if not c:
                continue
            for k in tot:
                tot[k] += c[k] * s["n_calls"]
            if s["site"] == "tt_bio/tenstorrent.py:4058":
                lever["F"] += c["F"] * s["n_calls"]
                lever["M"] += c["M"] * s["n_calls"]
        fold_ms = FOLD_S[model] * 1e3
        res["models"][model] = {
            "fold_ms": fold_ms, "softmax_F_ms": tot["F"], "softmax_R_ms": tot["R"],
            "softmax_M_ms": tot["M"],
            "softmax_share_of_fold": tot["F"] / fold_ms,
            "pred_slowdown_all_sites_M": (tot["M"] - tot["F"]) / fold_ms,
            "pred_slowdown_all_sites_R": (tot["R"] - tot["F"]) / fold_ms,
            "pred_slowdown_lever_only_M": (lever["M"] - lever["F"]) / fold_ms,
        }
        r = res["models"][model]
        print("%-13s fold %6.0f ms | softmax F %7.1f ms (%4.1f%% of fold) "
              "| M %7.1f ms -> +%5.2f%% fold | R -> +%5.2f%% | lever-only M -> +%5.2f%%"
              % (model, fold_ms, tot["F"], 100 * r["softmax_share_of_fold"], tot["M"],
                 100 * r["pred_slowdown_all_sites_M"], 100 * r["pred_slowdown_all_sites_R"],
                 100 * r["pred_slowdown_lever_only_M"]))

    out = os.path.join(ROOT, "perf/xmsoftmax/results/arm_cost_prediction.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n->", out)


if __name__ == "__main__":
    main()
