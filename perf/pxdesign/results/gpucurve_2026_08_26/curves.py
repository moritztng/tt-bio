#!/usr/bin/env python3
"""Batch curves for the PXDesign generator on both sides, and what they do to the published row.

GPU rungs come from the per-rung reports written by gpu_pxdesign_run.py on the two rented boxes;
Tenstorrent rungs from perf/newmodelcells/batchcurve* on pc's p150a. Writes CURVES.json next to
this file. The GPU s/design is warm_median_cell_s / n_sample, the same quantity the published
b=1 cells publish.
"""
import glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
NMC = os.path.join(ROOT, "perf", "newmodelcells")
PAGE = os.path.join(ROOT, "site", "data", "perf-512aa.json")


def gpu_curve(box):
    rungs, util = [], {}
    log = os.path.join(HERE, box, "ladder.log")
    if os.path.exists(log):
        for line in open(log):
            m = re.match(r"\[sweep\] == b(\d+) rep0 .*util=([0-9.]+)% ([0-9.]+)W vram=([0-9.]+)GiB", line)
            if m:
                util[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    for f in glob.glob(os.path.join(HERE, box, "run_b*.json")):
        d = json.load(open(f))
        if "warm_median_cell_s" not in d:
            continue                      # rung still running when this was pulled
        n = d["n_sample"]
        u = util.get(n, (None, None, None))
        rungs.append({
            "batch": n,
            "s_per_design": round(d["warm_median_cell_s"] / n, 4),
            "cell_s": d["warm_median_cell_s"],
            "warm_n": d["warm_n"],
            "spread_pct": d["warm_spread_pct"],
            "device_s_per_design": round(d["warm_median_gen_device_s"] / n, 4),
            "feat_s": d["warm_median_gen_feat_s"],
            "peak_vram_GiB": round(d["peak_vram_alloc_B"] / 2 ** 30, 3),
            "util_pct": u[0], "power_W": u[1],
            "gpu_exclusive": d["gpu_exclusive"],
            "validation_ok": d["validation"]["ok"],
            "digests": d.get("digests"),
            "sample_diffusion_calls": (d["counts"] or {}).get("sample_diffusion"),
            "pxd_predict_calls": (d["counts"] or {}).get("pxd_predict"),
        })
    rungs.sort(key=lambda r: r["batch"])
    if rungs:
        b1 = rungs[0]["s_per_design"]
        for r in rungs:
            r["amortisation_x"] = round(b1 / r["s_per_design"], 4)
    return rungs


def tt_curve():
    fit = json.load(open(os.path.join(NMC, "batchcurve", "FIT_400.json")))
    rungs = {r["batch"]: {"batch": r["batch"], "fitted_s_per_design": r["s_per_design"],
                          "distinct": r["distinct"]} for r in fit["rungs"]}
    for f in glob.glob(os.path.join(NMC, "batchcurve400", "c400_n*.json")):
        d = json.load(open(f))
        r = rungs.setdefault(d["n_sample_per_call"], {"batch": d["n_sample_per_call"]})
        r["measured_s_per_design"] = d["warm_median_s_per_design"]
        r["measured_spread_pct"] = d["warm_spread_pct_per_design"]
        r["measured_warm_n"] = d["warm_n"]
        r["distinct"] = d["all_designs_distinct"]
    out = [rungs[k] for k in sorted(rungs)]
    b1 = out[0]
    base = b1.get("measured_s_per_design") or b1["fitted_s_per_design"]
    for r in out:
        s = r.get("measured_s_per_design") or r["fitted_s_per_design"]
        r["amortisation_x"] = round(base / s, 4)
        if "measured_s_per_design" in r:
            r["fit_error_pct"] = round(100 * (r["fitted_s_per_design"] - r["measured_s_per_design"])
                                       / r["measured_s_per_design"], 3)
    return out


def chunk_check():
    """max_parallel_samples 8 with n_sample 32: is the chunk ceiling the batch ceiling?"""
    pts = {}
    for f in glob.glob(os.path.join(NMC, "batchchunk", "mps8_s*_n32.json")):
        d = json.load(open(f))
        warm = [x["s_per_design"] for x in d["designs"] if not x.get("cold")]
        pts[d["n_step"]] = statistics.median(warm) * d["n_sample_per_call"]
    if len(pts) < 2:
        return None
    (n0, c0), (n1, c1) = sorted(pts.items())
    P = (c1 - c0) / (n1 - n0)
    F = c0 - n0 * P
    return {"n_sample": 32, "max_parallel_samples": 8, "F_s": round(F, 4),
            "P_s_per_step": round(P, 6), "fit_from_n_step": [n0, n1],
            "s_per_design_at_400": round((F + 400 * P) / 32, 4)}


def main():
    page = json.load(open(PAGE))
    px = next(m for m in page["design"]["models"] if m["id"] == "pxdesign")
    cells = {k: v["s_per_design"] for k, v in px["cells"].items()}
    tt, h200, b200 = tt_curve(), gpu_curve("h200"), gpu_curve("b200")

    def best(rungs, key):
        return max(rungs, key=lambda r: r["amortisation_x"])

    out = {"fixture": px["target"], "n_step": px["sampling_steps"],
           "published_cells_s_per_design": cells,
           "tt": {"host": "pc", "card": "p150a physical 0", "label": "PROVISIONAL-ON-PC-CARD0",
                  "rungs": tt, "best": best(tt, "amortisation_x")["batch"],
                  "chunk_check": chunk_check()},
           "h200": {"rungs": h200, "best": best(h200, "amortisation_x")["batch"] if h200 else None},
           "b200": {"rungs": b200, "best": best(b200, "amortisation_x")["batch"] if b200 else None}}

    tt_a = best(tt, "amortisation_x")["amortisation_x"]
    tt_s = cells["p150a"] / tt_a
    srv = {"cards_galaxy": 32, "gpus_dgx": 8,
           "tt_best_batch_s_per_design": round(tt_s, 4), "tt_amortisation_x": tt_a}
    for box in ("h200", "b200"):
        if not out[box]["rungs"]:
            continue
        a = best(out[box]["rungs"], "amortisation_x")["amortisation_x"]
        s = cells[box] / a
        srv[box] = {"amortisation_x": a, "best_batch_s_per_design": round(s, 4),
                    "per_accelerator_x_b1": round(cells[box] / cells["p150a"], 4),
                    "per_accelerator_x_best_batch": round(s / tt_s, 4),
                    "per_server_x_b1": round((32 / cells["p150a"]) / (8 / cells[box]), 4),
                    "per_server_x_best_batch": round((32 / tt_s) / (8 / s), 4)}
    out["server"] = srv
    json.dump(out, open(os.path.join(HERE, "CURVES.json"), "w"), indent=2)
    print(json.dumps(out["server"], indent=2))
    for box in ("tt", "h200", "b200"):
        print("\n" + box)
        for r in out[box]["rungs"]:
            print("  ", {k: v for k, v in r.items() if k != "digests"})
    if out["tt"]["chunk_check"]:
        print("\nchunk", out["tt"]["chunk_check"])


main()
