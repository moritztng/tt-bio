"""RFD4's fair ratio against the B200, from the evidence on both sides. CPU only.

    python perf/rfd4/fair/verdict.py --runs <dir>/runs.jsonl [...] \
        [--gpu state/rfd4/evidence/gpu-b200.jsonl] [--state /home/moritz/.coworker/state/rfd4] [--dry]

Reads the TT runs `run.py` wrote and keeps, for the evidence file, only the lines the protocol
admits: warm measure batches on a p150a, not discarded (clock, contention, chunking), from a clean
tree whose sha is on tt-bio main. Per size it takes, among the runs with >= 5 such batches on the
newest main sha measured, the fastest arm (batch width, lane width), and reports its median designs/s
with min and max.

The B200 side is recomputed from `gpu-b200.jsonl` the way the gate does: warm lines grouped by
(size, arm, batch), the median per group, and per size the fastest group (>= 5 lines preferred,
fewer flagged). The ratio is PROTOCOL section 7:

    ratio = (32 x designs_s_p150a / 160000) / (8 x designs_s_b200 / 562154)

Writes `evidence/tt-p150a.jsonl`, `FAIR.md`, `VERDICT.json` and `rows/rfd4-fair-tt.json` under
`--state`, or with `--dry` prints what it would write.
"""
import argparse
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

SIZES = (128, 256, 512)
# Realistic binder sizes (rfd4-size-typical, state/rfd4/SIZE-TYPICAL.md): reported in their own table, never
# part of the gate, which stays defined on SIZES.
TYPICAL = (192, 224, 320)
BAR = 3.00
MIN_BATCHES = 5
GALAXY_CHIPS, GALAXY_USD = 32, 160000
DGX_GPUS, DGX_USD = 8, 562154
# The same eight B200s in an HGX chassis (rfd4-gpu-fair2, state/rfd4/GPU-AUDIT.md): every ratio/$ divides by this.
HGX_USD = 384228
HGX_FACTOR = DGX_USD / HGX_USD  # 1.4631
PER_CHIP = (GALAXY_CHIPS / GALAXY_USD) / (DGX_GPUS / DGX_USD)  # 14.054


def ratio(tt, gpu):
    return (GALAXY_CHIPS * tt / GALAXY_USD) / (DGX_GPUS * gpu / DGX_USD)


def r_sys(tt, gpu):
    """Raw throughput, 32 chips against 8 GPUs, no prices. "Matches a DGX B200" is r_sys >= 1, i.e. ratio >= 3.5135."""
    return GALAXY_CHIPS * tt / (DGX_GPUS * gpu)


def crossing(cells):
    """Token count where r_sys crosses 1.00, interpolated linearly in log(r_sys) between the two measured sizes
    that bracket it. None when no adjacent pair brackets 1.00."""
    import math
    pts = sorted((s, c["r_sys"]) for s, c in cells.items() if c.get("r_sys"))
    for (s0, r0), (s1, r1) in zip(pts, pts[1:]):
        if (r0 - 1) * (r1 - 1) <= 0 and r0 != r1:
            f = math.log(r0) / (math.log(r0) - math.log(r1))
            return {"tokens": round(s0 + f * (s1 - s0)), "between": [s0, s1], "r_sys": [r0, r1]}
    return None


HISTORY = "## Earlier verdicts"


def carry_history(path, fair, written):
    """The new FAIR.md, with every earlier verdict kept under one history heading.

    A fair number is evidence: rerunning this script must not delete the verdict it supersedes
    (it did, 2026-10-23, and the history had to be pasted back from a tool log).
    """
    if not path.exists():
        return fair
    old = path.read_text()
    head, _, tail = old.partition(HISTORY)
    older = tail.strip()
    block = [HISTORY, "", f"### Superseded at {written}", "", head.strip()]
    if older:
        block += ["", older]
    return fair.rstrip() + "\n\n" + "\n".join(block) + "\n"


def jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def on_main(sha, repo):
    r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=repo, capture_output=True)
    return r.returncode == 0


def admit(r, repo):
    """Why a TT line is not evidence, or None when it is."""
    if r.get("kind") != "warm" or r.get("phase") != "measure":
        return "not a warm measure batch"
    if r.get("part") != "p150a":
        return f"part {r.get('part')}, not p150a"
    if r.get("discard"):
        return r["discard"]
    if r.get("tree_dirty"):
        return "dirty tree"
    if "trunk_passes_counted" in r and r["trunk_passes_counted"] != r["trunk_passes"]:
        return f"work_mismatch: {r['trunk_passes_counted']} trunk passes counted, {r['trunk_passes']} scheduled"
    if not on_main(r["sha"], repo):
        return f"sha {r['sha'][:9]} is not on main"
    return None


def tt_side(paths, repo):
    lines, rejected = [], defaultdict(int)
    for p in paths:
        for r in jsonl(p):
            why = admit(r, repo)
            if why is None:
                lines.append(r)
            elif r.get("kind") == "warm" and r.get("phase") == "measure":
                rejected[why.split(":")[0]] += 1
    per = {}
    for size in SIZES + TYPICAL:
        runs = defaultdict(list)
        for r in lines:
            if r["size"] == size:
                runs[r["raw"]].append(r)
        full = [v for v in runs.values() if len(v) >= MIN_BATCHES]
        if not full:
            continue
        sha = max(full, key=lambda v: max(r["t"] for r in v))[0]["sha"]  # the newest main measured
        v = max((v for v in full if v[0]["sha"] == sha), key=lambda v: statistics.median(r["designs_per_s"] for r in v))
        d = [r["designs_per_s"] for r in v]
        per[size] = {"designs_per_s": round(statistics.median(d), 5), "min": round(min(d), 5), "max": round(max(d), 5),
                     "batches": len(v), "batch": v[0]["batch"], "lane_width": v[0].get("lane_width") or v[0]["batch"], "sha": v[0]["sha"], "host": v[0]["host"],
                     "card": v[0]["card"], "board_type": v[0].get("board_type"), "raw": v[0]["raw"],
                     "aiclk_mhz": sorted(r["aiclk_mhz"] for r in v),
                     "aiclk_min_mhz": min(r["aiclk_min_mhz"] for r in v),
                     "foreign_cores_median": max(r["foreign_cores_median"] or 0 for r in v),
                     "host_load_per_core_median": max(r["host_load_per_core_median"] or 0 for r in v),
                     "power_w": [r.get("power_w") for r in v],
                     "date": max(r["t"] for r in v), "lines": v}
    return per, dict(rejected)


def host_control(path, repo, ref, cells):
    """A control cell: an older main tree re-measured on this run's card, so the host/firmware difference to the
    card that first measured that tree (`ref`) is separated from the code gain since. Never a fair cell."""
    v = [r for r in jsonl(path) if admit(r, repo) is None]
    if len(v) < MIN_BATCHES:
        return {"status": "short", "n_clean": len(v), "raw": str(path)}
    d = [r["designs_per_s"] for r in v]
    c = {"size": v[0]["size"], "batch": v[0]["batch"], "lane_width": v[0].get("lane_width") or v[0]["batch"],
         "sha": v[0]["sha"], "host": v[0]["host"], "card": v[0]["card"], "n_clean": len(v),
         "designs_per_s": round(statistics.median(d), 5), "min": round(min(d), 5), "max": round(max(d), 5),
         "aiclk_mhz": sorted(r["aiclk_mhz"] for r in v), "raw": str(path), "reference": ref}
    if ref and ref.get("designs_per_s"):
        c["host_factor"] = round(c["designs_per_s"] / ref["designs_per_s"], 3)
    main = cells.get(c["size"])
    if main:
        c["code_gain"] = round(main["designs_per_s"] / c["designs_per_s"], 3)
    return c


def gpu_side(path):
    groups = defaultdict(list)
    for r in jsonl(path):
        if r.get("kind") == "warm":
            groups[(r["size"], r.get("arm"), r.get("batch"))].append(r)
    per = {}
    for (size, arm, b), v in groups.items():
        d = [r["designs"] / r["wall_s"] for r in v]
        cand = {"designs_per_s": round(statistics.median(d), 5), "min": round(min(d), 5), "max": round(max(d), 5),
                "arm": arm, "batch": b, "batches": len(v), "gpu": v[0].get("gpu"), "host": v[0].get("host"),
                "sha": v[0].get("sha"), "date": max(r.get("t", "") for r in v), "session": v[0].get("session"),
                "sm_mhz": statistics.median(r.get("sm_mhz_busy_median") or r.get("sm_mhz") or 0 for r in v),
                "short": len(v) < MIN_BATCHES}
        cur = per.get(size)
        if cur is None or (cur["short"], -cur["designs_per_s"]) > (cand["short"], -cand["designs_per_s"]):
            per[size] = cand
    return per


def evidence_line(r):
    keep = ("kind", "size", "designs", "wall_s", "batch", "part", "host", "card", "aiclk_mhz", "aiclk_min_mhz",
            "aiclk_samples", "power_w", "power_w_peak", "sha", "raw", "seed", "t", "board_type",
            "foreign_cores_median", "foreign_device_holders", "host_load_per_core_median", "recipe", "steps",
            "trunk_passes", "trunk_passes_counted", "trunk_blocks_counted", "decoded_counted", "digest",
            "featurize_s", "design_s", "write_s", "lane_width")
    return {"arm": "tt-bio main", **{k: r.get(k) for k in keep}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--state", default="/home/moritz/.coworker/state/rfd4")
    ap.add_argument("--gpu", help="default <state>/evidence/gpu-b200.jsonl")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--deck", help="default <state>/../biodeck/src/extra_cells.json")
    ap.add_argument("--firmware", required=True, help="the card's firmware bundle during the runs, e.g. 19.8.1 (130-core)")
    ap.add_argument("--row", help="also write rows/<ROW>.json: every measured size, gated and realistic, with r_sys")
    ap.add_argument("--control", help="runs.jsonl of a host-control cell (an older main tree on this card)")
    ap.add_argument("--control-ref", type=json.loads, default=None,
                    help='JSON: that tree as first measured, e.g. {"designs_per_s": 0.04408, "what": "pc card 0"}')
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    state = Path(a.state)
    gpu_path = a.gpu or state / "evidence" / "gpu-b200.jsonl"

    tt, rejected = tt_side(a.runs, a.repo)
    gpu = gpu_side(gpu_path)
    sizes, typical = {}, {}
    for s in SIZES + TYPICAL:
        t, g = tt.get(s), gpu.get(s)
        cell = {"tt": t and {k: v for k, v in t.items() if k != "lines"}, "b200": g}
        if t and g:
            cell["ratio"] = round(ratio(t["designs_per_s"], g["designs_per_s"]), 3)
            cell["r_sys"] = round(r_sys(t["designs_per_s"], g["designs_per_s"]), 3)
            cell["p150a_share_of_b200"] = round(t["designs_per_s"] / g["designs_per_s"], 4)
        (sizes if s in SIZES else typical)[s] = cell
    ratios = [c.get("ratio") for c in sizes.values()]
    if all(r is not None for r in ratios):
        verdict = "PASS" if min(ratios) >= BAR else "FAIL"
    else:
        verdict = "INCOMPLETE"
    gpu_short = any(g and g["short"] for g in gpu.values())

    out = {"verdict": verdict, "bar": BAR, "per_chip_factor": round(PER_CHIP, 3),
           "formula": "(32 x designs_s_p150a / 160000) / (8 x designs_s_b200 / 562154)",
           "p150a_firmware": a.firmware,
           "sizes": sizes, "realistic_binder_sizes": typical,
           "r_sys_formula": "32 x designs_s_p150a / (8 x designs_s_b200) = ratio / 3.5135; >= 1.00 matches a DGX B200",
           "r_sys_crossing": crossing({**sizes, **typical}), "tt_rejected_measure_lines": rejected,
           "host_control": a.control and host_control(a.control, a.repo, a.control_ref, tt),
           "b200_source": str(gpu_path), "b200_complete": all(s in gpu for s in SIZES) and not gpu_short,
           "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ev = [evidence_line(r) for s in SIZES + TYPICAL if s in tt for r in tt[s]["lines"]]
    gated = {s: v for s, v in tt.items() if s in SIZES}
    fair = fair_md(out)
    row = {"status": "done" if verdict in ("PASS", "FAIL") and out["b200_complete"] else "running",
           "part": "p150a", "firmware": a.firmware, "host": sorted({t["host"] for t in gated.values()}),
           "card": sorted({t["card"] for t in gated.values()}), "sizes": list(SIZES),
           "designs_per_s": {s: tt[s]["designs_per_s"] for s in gated},
           "aiclk_mhz": {s: tt[s]["aiclk_mhz"] for s in gated},
           "host_load": {s: {"foreign_cores_median_max": tt[s]["foreign_cores_median"],
                             "load_per_core_median_max": tt[s]["host_load_per_core_median"]} for s in gated},
           "main_sha": {s: tt[s]["sha"] for s in gated},
           "b200_reference": {s: gpu[s] for s in gpu if s in SIZES},
           "ratio": {s: c.get("ratio") for s, c in sizes.items()}, "verdict": verdict,
           "updated": out["written"]}
    allm = {**sizes, **typical}
    meas = {s: tt[s] for s in sorted(allm) if s in tt}
    full = {"status": "done" if out["b200_complete"] and all(allm[s].get("ratio") for s in meas) else "running",
            "part": "p150a", "fw": a.firmware, "host": sorted({t["host"] for t in meas.values()}),
            "card": sorted({t["card"] for t in meas.values()}), "sizes": list(meas),
            "designs_per_s": {s: t["designs_per_s"] for s, t in meas.items()},
            "batch": {s: t["batch"] for s, t in meas.items()},
            "lane_width": {s: t["lane_width"] for s, t in meas.items()},
            "n_clean": {s: t["batches"] for s, t in meas.items()},
            "aiclk_mhz": {s: t["aiclk_mhz"] for s, t in meas.items()},
            "aiclk_min_mhz": {s: t["aiclk_min_mhz"] for s, t in meas.items()},
            "host_load": {s: {"foreign_cores_median_max": t["foreign_cores_median"],
                              "load_per_core_median_max": t["host_load_per_core_median"]} for s, t in meas.items()},
            "main_sha": {s: t["sha"] for s, t in meas.items()},
            "raw": {s: t["raw"] for s, t in meas.items()},
            "b200_reference": {s: allm[s]["b200"] for s in meas},
            "ratio": {s: allm[s].get("ratio") for s in meas},
            "ratio_hgx": {s: allm[s].get("ratio") and round(allm[s]["ratio"] / HGX_FACTOR, 3) for s in meas},
            "r_sys": {s: allm[s].get("r_sys") for s in meas},
            "r_sys_crossing": out["r_sys_crossing"], "host_control": out["host_control"], "verdict": verdict, "verdict_written": out["written"]}
    if a.dry:
        print(json.dumps(out, indent=1, default=str))
        print(json.dumps(full, indent=1, default=str))
        print(fair)
        return
    (state / "evidence" / "tt-p150a.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ev))
    (state / "VERDICT.json").write_text(json.dumps(out, indent=1) + "\n")
    (state / "FAIR.md").write_text(carry_history(state / "FAIR.md", fair, out["written"]))
    (state / "rows" / "rfd4-fair-tt.json").write_text(json.dumps(row, indent=1) + "\n")
    if a.row:
        (state / "rows" / f"{a.row}.json").write_text(json.dumps(full, indent=1) + "\n")
    if verdict in ("PASS", "FAIL"):
        deck = Path(a.deck or state.parent / "biodeck" / "src" / "extra_cells.json")
        cells = {k: v for k, v in json.loads(deck.read_text()).items() if not k.startswith("rfd4.")}
        cells.update(deck_cells(out, gpu_path))
        deck.write_text(json.dumps(cells, indent=1) + "\n")
    print(fair)


def deck_cells(v, gpu_path):
    """The rfd4.* deck cells, shaped like rfd3.*: one measured cell per size and side, never a projection."""
    cells = {}
    for s, c in v["sizes"].items():
        t, g = c["tt"], c["b200"]
        cells[f"rfd4.p150a.{s}"] = {
            "s": round(1 / t["designs_per_s"], 3),
            "unit": f"s/design (warm median of {t['batches']} batches at batch {t['batch']}, "
                    f"{t['min']}-{t['max']} designs/s)",
            "n": t["batches"], "aiclk_mhz": statistics.median(t["aiclk_mhz"]), "batch": t["batch"],
            "versions": f"tt-bio main {t['sha'][:9]}, RFdiffusion4 thinking_medium, 60 steps / 180 trunk passes, "
                        f"pdl1-{s} frozen spec, ttnn 0.68.0, firmware {v['p150a_firmware']}",
            "when": t["date"], "card": f"{t['host']}:{t['card']} {t['board_type']}",
            "aiclk_samples": f"min {t['aiclk_min_mhz']} MHz in-run, foreign cores median <= {t['foreign_cores_median']}",
            "source": f"{t['raw']}/runs.jsonl (perf/rfd4/fair/run.py)",
            "ratio_vs_b200": c["ratio"]}
        cells[f"rfd4.b200.{s}"] = {
            "s": round(1 / g["designs_per_s"], 3),
            "unit": f"s/design (warm median of {g['batches']} batches at batch {g['batch']}, arm {g['arm']})",
            "n": g["batches"], "clock_mhz": g["sm_mhz"], "batch": g["batch"],
            "versions": f"RFdiffusion4 upstream, arm {g['arm']}, thinking_medium, pdl1-{s}",
            "when": g["date"], "box": f"1x {g['gpu']} (session {g['session']})", "source": str(gpu_path)}
    return cells


def fair_md(v):
    L = ["# RFD4 fair number: one p150a against one B200", "",
         f"Written by `perf/rfd4/fair/verdict.py` at {v['written']}. Protocol `state/rfd4/PROTOCOL.md` (frozen).",
         f"Ratio per size = {v['formula']}; per chip that is {v['per_chip_factor']}x, so {v['bar']:.2f}x needs "
         f"one p150a at >= {v['bar'] / v['per_chip_factor']:.4f}x one B200.", "",
         f"**Verdict: {v['verdict']}** against the {v['bar']:.2f}x bar. p150a firmware: {v['p150a_firmware']}.", "",
         "| size | p150a designs/s (min-max, n) | batch | AICLK MHz (per batch) | foreign cores | card, host | main sha "
         "| B200 designs/s (arm, batch, n) | ratio |", "|---|---|---|---|---|---|---|---|---|"]
    for s, c in v["sizes"].items():
        t, g = c["tt"], c["b200"]
        b = t and (t["batch"] if t.get("lane_width", t["batch"]) == t["batch"] else f"{t['batch']} (lanes of {t['lane_width']})")
        tc = (f"{t['designs_per_s']} ({t['min']}-{t['max']}, {t['batches']})", b,
              "/".join(str(x) for x in t["aiclk_mhz"]), t["foreign_cores_median"], f"{t['card']}, {t['host']}",
              t["sha"][:9]) if t else ("not measured", "", "", "", "", "")
        gc = (f"{g['designs_per_s']} ({g['arm']}, b{g['batch']}, {g['batches']}"
              f"{', short' if g['short'] else ''})") if g else "not published"
        L.append(f"| {s} | " + " | ".join(str(x) for x in tc) + f" | {gc} | {c.get('ratio', '')} |")
    L += typical_md(v)
    L += ["", f"B200 side recomputed from `{v['b200_source']}`"
          + ("" if v["b200_complete"] else "; it is incomplete (missing sizes or fewer than 5 batches), "
             "so every ratio here is against what it has so far") + ".",
          f"TT measure lines rejected from the evidence: {v['tt_rejected_measure_lines'] or 'none'}.",
          *control_md(v.get("host_control")),
          "", "## How these ratios can be quoted", "",
          f"Chassis: every ratio/$ prices the B200 side as a DGX B200 at ${DGX_USD:,}. The same eight B200s in an HGX "
          f"chassis cost ${HGX_USD:,} (rfd4-gpu-fair2); on that basis every ratio divides by {HGX_FACTOR:.4f}. "
          "A ratio quoted without its chassis is not quotable.",
          "Scale: 32 chips and 8 GPUs are each one measured device times the count. The Galaxy's multi-chip "
          "efficiency is owned by rfd4-bh-scaling (`state/rfd4/SCALING.md`); until it reports, x32 is an "
          "extrapolation."]
    return "\n".join(L) + "\n"


def control_md(c):
    if not c:
        return []
    if c.get("status") == "short":
        return ["", f"Host control: {c['n_clean']} clean batches, short of {MIN_BATCHES}; no host factor."]
    ref = c.get("reference") or {}
    L = ["", "## Host control (not a fair cell)", "",
         f"Main {c['sha'][:9]} re-measured at {c['size']} b{c['batch']} (lanes of {c['lane_width']}) on "
         f"{c['host']} card {c['card']}: {c['designs_per_s']} designs/s ({c['min']}-{c['max']}, {c['n_clean']}), "
         f"AICLK {'/'.join(str(x) for x in c['aiclk_mhz'])}."]
    if "host_factor" in c:
        L.append(f"The same tree read {ref['designs_per_s']} on {ref.get('what', 'the reference card')}, so this card "
                 f"and firmware run it at {c['host_factor']}x.")
    if "code_gain" in c:
        L.append(f"Main as measured above is {c['code_gain']}x this tree on the same card: the code gain with the "
                 "host held fixed.")
    return L


def typical_md(v):
    """The realistic-binder-size table and the raw-throughput view, beside (not inside) the gated table."""
    L = ["", "## Realistic binder sizes (not part of the gate)", "",
         "PD-L1 (115 residues) with a fixed binder of 77 / 109 / 205 residues; why these sizes is in "
         "`state/rfd4/SIZE-TYPICAL.md`. Same protocol, same per-chip arithmetic. `R_sys` is the raw throughput of "
         "32 p150a chips against 8 B200 GPUs (ratio / 3.5135): 1.00 is \"matches a DGX B200\". Both sides are "
         "extrapolated from one measured chip or GPU.", "",
         "| size | binder | p150a designs/s (min-max, n) | batch | AICLK MHz | foreign cores | main sha "
         "| B200 designs/s (arm, batch, n) | ratio/$ | R_sys |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, c in v["realistic_binder_sizes"].items():
        t, g = c["tt"], c["b200"]
        b = t and (t["batch"] if t.get("lane_width", t["batch"]) == t["batch"] else f"{t['batch']} (lanes of {t['lane_width']})")
        tc = (f"{t['designs_per_s']} ({t['min']}-{t['max']}, {t['batches']})", b,
              "/".join(str(x) for x in t["aiclk_mhz"]), t["foreign_cores_median"], t["sha"][:9]) if t \
            else ("not measured", "", "", "", "")
        gc = (f"{g['designs_per_s']} ({g['arm']}, b{g['batch']}, {g['batches']}"
              f"{', short' if g['short'] else ''})") if g else "not measured"
        L.append(f"| {s} | {s - 115} | " + " | ".join(str(x) for x in tc)
                 + f" | {gc} | {c.get('ratio', '')} | {c.get('r_sys', '')} |")
    allc = {**v["sizes"], **v["realistic_binder_sizes"]}
    L += ["", "R_sys at every measured size: " + ", ".join(f"{s} {c['r_sys']}" for s, c in sorted(allc.items())
                                                      if c.get("r_sys")) + "."]
    x = v.get("r_sys_crossing")
    L.append(f"R_sys crosses 1.00 at about **{x['tokens']} tokens** (log-linear between the measured {x['between'][0]} "
             f"and {x['between'][1]}, R_sys {x['r_sys'][0]} and {x['r_sys'][1]}); an interpolation, not a measurement."
             if x else "No pair of adjacent measured sizes brackets R_sys = 1.00.")
    return L


if __name__ == "__main__":
    main()
