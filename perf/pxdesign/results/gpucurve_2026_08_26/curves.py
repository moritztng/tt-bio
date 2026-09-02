#!/usr/bin/env python3
"""Batch curves for the PXDesign generator on both sides, and what they do to the published row.

GPU rungs come from the per-rung reports written by gpu_pxdesign_run.py on the two rented boxes;
Tenstorrent rungs from perf/newmodelcells/<prefix>batchcurve*, where --tt_prefix picks the box:
the default qb2_ is the qb2 p300c re-measurement, --tt_prefix '' reproduces the original pc p150a
run, whose seconds were never publishable. Writes CURVES.json next to this file. The GPU s/design is warm_median_cell_s / n_sample, the same quantity the published
b=1 cells publish.
"""
import argparse, glob, json, os, re, statistics, sys

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


# Which run publishes a rung. On 2026-09-02 another worker ran a CPU parity reference on card 1 from
# 00:38 to 02:15, outside the benchlock, burning 7.5 of the box"s 16 cores. Its measured cost here is
# batch-dependent -- 9.6 % at b=2, 5.8 % at b=4, 2.6 % at b=16 -- so it cannot be corrected for, only
# avoided: a rung publishes a run whose WARM rounds were all timed on a quiet box.
#
# The per-round loadavg in each JSON decides that, not the wall-clock window the run fell in. The
# window got it wrong in both directions. d400_n8r5 started at 00:23 and was booked as quiet, but its
# rounds 2 and 3 ran at loadavg 8.50 and 9.84; c400_n64 was booked as loaded, but only its discarded
# cold round ran at 8.96 and both warm rounds ran at 1.00 and 1.08.
QUIET_LOADAVG = 2.0


def _warm(d):
    return [r for r in d["designs"] if not r.get("cold")]


def _peak_loadavg(d):
    la = [float(r["loadavg"][0]) for r in _warm(d) if r.get("loadavg")]
    return max(la) if la else float("inf")


def tt_curve(prefix):
    fit = json.load(open(os.path.join(NMC, prefix + "batchcurve", "FIT_400.json")))
    rungs = {r["batch"]: {"batch": r["batch"], "fitted_s_per_design": r["s_per_design"],
                          "distinct": r["distinct"]} for r in fit["rungs"]}
    prov = {}
    d400 = os.path.join(NMC, prefix + "batchcurve400")
    cand = {}
    for f in sorted(glob.glob(os.path.join(d400, "*400_n*.json"))):
        d = json.load(open(f))
        if not d.get("warm_n"):
            continue                       # rung still running when this was pulled
        cand.setdefault(d["n_sample_per_call"], []).append((f, d))

    # A quiet arm beats a loaded one however many rounds the loaded one has. Among equally quiet arms
    # more warm rounds win, and a tie there goes to the tighter spread, which is the arm that settled.
    # d400_n8r5 never did: it ramps 159.1 -> 145.1 s across five rounds and only its last round meets
    # ctl400_n8"s three consecutive 145.0 s.
    def rank(fd):
        d = fd[1]
        return (_peak_loadavg(d) <= QUIET_LOADAVG, d["warm_n"], -d["warm_spread_pct_per_design"])

    for b in sorted(cand):
        f, d = max(cand[b], key=rank)
        r = rungs.setdefault(b, {"batch": b})
        r["measured_s_per_design"] = d["warm_median_s_per_design"]
        r["measured_spread_pct"] = d["warm_spread_pct_per_design"]
        r["measured_warm_n"] = d["warm_n"]
        r["measured_from"] = os.path.basename(f)
        r["peak_warm_loadavg"] = _peak_loadavg(d)
        r["regime"] = "quiet" if r["peak_warm_loadavg"] <= QUIET_LOADAVG else "loaded"
        r["distinct"] = d["all_designs_distinct"]
        others = [(os.path.basename(g), e["warm_median_s_per_design"], _peak_loadavg(e))
                  for g, e in cand[b] if g != f]
        if others:
            r["not_published"] = [{"file": g, "s_per_design": s, "peak_warm_loadavg": l}
                                  for g, s, l in others]
        prov = {"host": d["host"], "card": d["card"], "ttnn": d["ttnn"], "git_head": d["git_head"],
                "grid": d.get("grid"), "diffusion_fp32": d.get("diffusion_fp32")}
    out = [rungs[k] for k in sorted(rungs)]
    base = out[0].get("measured_s_per_design") or out[0]["fitted_s_per_design"]
    for r in out:
        sd = r.get("measured_s_per_design") or r["fitted_s_per_design"]
        r["amortisation_x"] = round(base / sd, 4)
        if "measured_s_per_design" in r:
            r["fit_error_pct"] = round(100 * (r["fitted_s_per_design"] - r["measured_s_per_design"])
                                       / r["measured_s_per_design"], 3)
    return out, prov


def seed_check(prefix):
    """Round 4 repeats round 0's seed, so its coordinate digest must repeat. pc's fp32 rate on this
    shape was 0/14, so a clean rate here is a finding in its own right."""
    hits = []
    for f in sorted(glob.glob(os.path.join(NMC, prefix + "batchcurve400", "*.json"))) + \
            sorted(glob.glob(os.path.join(NMC, prefix + "batchcurve", "*.json"))):
        d = json.load(open(f))
        if "designs" not in d:
            continue
        by_seed = {}
        for r in d["designs"]:
            by_seed.setdefault(r["seed"], []).append(r["coord_sha16"])
        for seed, digs in sorted(by_seed.items()):
            if len(digs) > 1:
                hits.append({"file": os.path.basename(f), "batch": d["n_sample_per_call"],
                             "n_step": d["n_step"], "seed": seed, "digests": digs,
                             "reproduced": len(set(digs)) == 1})
    if not hits:
        return None
    return {"pairs": hits, "reproduced": sum(h["reproduced"] for h in hits), "total": len(hits)}


def chunk_check(prefix):
    """max_parallel_samples 8 with n_sample 32: is the chunk ceiling the batch ceiling?

    A real 400-step run wins outright where one exists. The two-point fit from n_step 8 and 24 is
    only a fallback, because on qb2 that fit overshoots the measured 400-step rung by 29 % at b=8
    (state doc section 3): short runs on this box sit in a noise floor the extrapolation inherits.
    """
    real = os.path.join(NMC, prefix + "batchchunk", "mps8_c400_n32.json")
    if os.path.exists(real):
        d = json.load(open(real))
        if d.get("warm_n"):
            return {"n_sample": d["n_sample_per_call"],
                    "max_parallel_samples": d["max_parallel_samples"],
                    "measured_at_n_step": d["n_step"], "warm_n": d["warm_n"],
                    "spread_pct": d["warm_spread_pct_per_design"],
                    "s_per_design_at_400": d["warm_median_s_per_design"]}
    pts = {}
    for f in glob.glob(os.path.join(NMC, prefix + "batchchunk", "mps8_s*_n32.json")):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt_prefix", default="qb2_",
                    help="perf/newmodelcells/<prefix>batchcurve* to read the TT rungs from")
    ap.add_argument("--board", default="p300c", help="board type of the card that ran them")
    a = ap.parse_args()
    page = json.load(open(PAGE))
    px = next(m for m in page["design"]["models"] if m["id"] == "pxdesign")
    cells = {k: v["s_per_design"] for k, v in px["cells"].items()}
    tt, prov = tt_curve(a.tt_prefix)
    h200, b200 = gpu_curve("h200"), gpu_curve("b200")

    def best(rungs, key):
        return max(rungs, key=lambda r: r["amortisation_x"])

    # The TT ladder carries a fitted s/design for every rung and a measured one only where a
    # 400-step run exists, and on this box the fit overshoots by up to 280 % (state doc section 3).
    # So the TT best batch may only be chosen among measured rungs, and a rung that is still fitted
    # is reported as such rather than quietly competing for the headline.
    def tt_best(rungs):
        m = [r for r in rungs if "measured_s_per_design" in r]
        if not m:
            sys.exit("curves.py: no measured 400-step TT rung; nothing here may be published")
        return max(m, key=lambda r: r["amortisation_x"])

    out = {"fixture": px["target"], "n_step": px["sampling_steps"],
           "published_cells_s_per_design": cells,
           "tt": {"host": prov.get("host"), "board": a.board,
                  "card": "%s physical %s" % (a.board, prov.get("card")),
                  "provenance": "on a %s board in %s, physical card %s"
                                % (a.board, prov.get("host"), prov.get("card")),
                  "ttnn": prov.get("ttnn"), "git_head": prov.get("git_head"),
                  "grid": prov.get("grid"), "diffusion_fp32": prov.get("diffusion_fp32"),
                  "rungs": tt, "best": tt_best(tt)["batch"],
                  "unmeasured_rungs": [r["batch"] for r in tt if "measured_s_per_design" not in r],
                  "loaded_rungs": [r["batch"] for r in tt if r.get("regime") == "loaded"],
                  "seed_check": seed_check(a.tt_prefix),
                  "chunk_check": chunk_check(a.tt_prefix)},
           "h200": {"rungs": h200, "best": best(h200, "amortisation_x")["batch"] if h200 else None},
           "b200": {"rungs": b200, "best": best(b200, "amortisation_x")["batch"] if b200 else None}}

    tt_a = tt_best(tt)["amortisation_x"]
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
    if out["tt"]["seed_check"]:
        sc = out["tt"]["seed_check"]
        print("\nseed  %d/%d repeated-seed pairs reproduced their digest" %
              (sc["reproduced"], sc["total"]))
        for h in sc["pairs"]:
            print("  ", h)


main()
