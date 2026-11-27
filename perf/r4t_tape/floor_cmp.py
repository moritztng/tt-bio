"""Per-tensor distance from the float64 reference of (a) a device gradient dump and (b) the same reference
in host float32: how far the device sits above what float32 arithmetic alone costs this network.

    python3 perf/r4t_tape/floor_cmp.py --dump OUT/fp32ex --ref64 REF.npz --ref32 REF32.npz [--out J]
"""
import argparse
import json
import numpy as np


def metrics(x, r):
    x, r = x.ravel().astype(np.float64), r.ravel().astype(np.float64)
    nr = np.linalg.norm(r)
    return float(np.linalg.norm(x - r) / max(nr, 1e-300)), float(x @ r / max(np.linalg.norm(x) * nr, 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref64", required=True)
    ap.add_argument("--ref32", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d, r64, r32 = np.load(a.dump + ".npz"), np.load(a.ref64), np.load(a.ref32)
    rows = {}
    for k in d.files:
        if not np.abs(r64[k]).max():
            continue
        dr, dc = metrics(d[k], r64[k])
        fr, fc = metrics(r32[k], r64[k])
        rows[k] = dict(dev_rel=dr, dev_cos=dc, f32_rel=fr, f32_cos=fc, ratio=dr / max(fr, 1e-300))
    med = lambda f: float(np.median([v[f] for v in rows.values()]))  # noqa: E731
    top = sorted(rows, key=lambda k: -rows[k]["ratio"])[:8]
    summ = {"n": len(rows), "dev_median": med("dev_rel"), "f32_median": med("f32_rel"), "ratio_median": med("ratio"),
            "dev_max": max(rows.items(), key=lambda kv: kv[1]["dev_rel"])[0],
            "f32_max": max(rows.items(), key=lambda kv: kv[1]["f32_rel"])[0],
            "f32_min_cos": min(v["f32_cos"] for v in rows.values()),
            "top_ratio": [(k, round(rows[k]["ratio"], 2), f"{rows[k]['dev_rel']:.2e}", f"{rows[k]['f32_rel']:.2e}") for k in top]}
    summ["dev_max"] = [summ["dev_max"], rows[summ["dev_max"]]["dev_rel"]]
    summ["f32_max"] = [summ["f32_max"], rows[summ["f32_max"]]["f32_rel"]]
    print(json.dumps(summ, indent=1))
    if a.out:
        json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
