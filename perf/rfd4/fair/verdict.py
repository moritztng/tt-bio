"""RFD4's fair ratio against the B200, from the evidence on both sides. CPU only.

    python perf/rfd4/fair/verdict.py --runs <dir>/runs.jsonl [...] \
        [--gpu state/rfd4/evidence/gpu-b200.jsonl] [--state /home/moritz/.coworker/state/rfd4] [--dry]

Reads the TT runs `run.py` wrote and keeps, for the evidence file, only the lines the protocol
admits: warm measure batches on a p150a, not discarded (clock, contention, chunking), from a clean
tree whose sha is on tt-bio main. Per size it takes the newest run with >= 5 such batches and
reports the median designs/s with min and max.

The B200 side is recomputed from `gpu-b200.jsonl` the way the gate does: warm lines grouped by
(size, arm, batch), the median per group, and per size the fastest group (>= 5 lines preferred,
fewer flagged). The ratio is PROTOCOL section 7:

    ratio = (32 x designs_s_p150a / 160000) / (8 x designs_s_b200 / 562154)

Writes `evidence/tt-p150a.jsonl`, `FAIR.md`, `VERDICT.json` and `rows/<--row>.json` under
`--state`, or with `--dry` prints what it would write. A FAIR.md from a different measurement is
kept below the new table as history.
"""
import argparse
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

SIZES = (128, 256, 512)
BAR = 3.00
MIN_BATCHES = 5
GALAXY_CHIPS, GALAXY_USD = 32, 160000
DGX_GPUS, DGX_USD = 8, 562154
PER_CHIP = (GALAXY_CHIPS / GALAXY_USD) / (DGX_GPUS / DGX_USD)  # 14.054


def ratio(tt, gpu):
    return (GALAXY_CHIPS * tt / GALAXY_USD) / (DGX_GPUS * gpu / DGX_USD)


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
    for size in SIZES:
        runs = defaultdict(list)
        for r in lines:
            if r["size"] == size:
                runs[r["raw"]].append(r)
        full = [v for v in runs.values() if len(v) >= MIN_BATCHES]
        if not full:
            continue
        v = max(full, key=lambda v: max(r["t"] for r in v))
        d = [r["designs_per_s"] for r in v]
        per[size] = {"designs_per_s": round(statistics.median(d), 5), "min": round(min(d), 5), "max": round(max(d), 5),
                     "batches": len(v), "batch": v[0]["batch"], "sha": v[0]["sha"], "host": v[0]["host"],
                     "card": v[0]["card"], "board_type": v[0].get("board_type"), "raw": v[0]["raw"],
                     "aiclk_mhz": sorted(r["aiclk_mhz"] for r in v),
                     "aiclk_min_mhz": min(r["aiclk_min_mhz"] for r in v),
                     "foreign_cores_median": max(r["foreign_cores_median"] or 0 for r in v),
                     "host_load_per_core_median": max(r["host_load_per_core_median"] or 0 for r in v),
                     "power_w": [r.get("power_w") for r in v],
                     "date": max(r["t"] for r in v), "lines": v}
    return per, dict(rejected)


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
            "trunk_passes", "featurize_s", "design_s", "write_s")
    return {"arm": "tt-bio main", **{k: r.get(k) for k in keep}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--state", default="/home/moritz/.coworker/state/rfd4")
    ap.add_argument("--gpu", help="default <state>/evidence/gpu-b200.jsonl")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--deck", help="default <state>/../biodeck/src/extra_cells.json")
    ap.add_argument("--firmware", required=True, help="the card's firmware bundle during the runs, e.g. 19.8.1 (130-core)")
    ap.add_argument("--row", default="rfd4-fair-tt", help="the campaign row whose rows/<row>.json this writes")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    state = Path(a.state)
    gpu_path = a.gpu or state / "evidence" / "gpu-b200.jsonl"

    tt, rejected = tt_side(a.runs, a.repo)
    gpu = gpu_side(gpu_path)
    sizes = {}
    for s in SIZES:
        t, g = tt.get(s), gpu.get(s)
        cell = {"tt": t and {k: v for k, v in t.items() if k != "lines"}, "b200": g}
        if t and g:
            cell["ratio"] = round(ratio(t["designs_per_s"], g["designs_per_s"]), 3)
            cell["p150a_share_of_b200"] = round(t["designs_per_s"] / g["designs_per_s"], 4)
        sizes[s] = cell
    ratios = [c.get("ratio") for c in sizes.values()]
    if all(r is not None for r in ratios):
        verdict = "PASS" if min(ratios) >= BAR else "FAIL"
    else:
        verdict = "INCOMPLETE"
    gpu_short = any(g and g["short"] for g in gpu.values())

    out = {"verdict": verdict, "bar": BAR, "per_chip_factor": round(PER_CHIP, 3),
           "formula": "(32 x designs_s_p150a / 160000) / (8 x designs_s_b200 / 562154)",
           "p150a_firmware": a.firmware,
           "sizes": sizes, "tt_rejected_measure_lines": rejected,
           "b200_source": str(gpu_path), "b200_complete": all(s in gpu for s in SIZES) and not gpu_short,
           "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ev = [evidence_line(r) for s in SIZES if s in tt for r in tt[s]["lines"]]
    fair = fair_md(out, state / "FAIR.md", state / "VERDICT.json")
    row = {"status": "done" if verdict in ("PASS", "FAIL") and out["b200_complete"] else "running",
           "part": "p150a", "fw": a.firmware, "host": sorted({t["host"] for t in tt.values()}),
           "card": sorted({t["card"] for t in tt.values()}), "sizes": list(SIZES),
           "designs_per_s": {s: tt[s]["designs_per_s"] for s in tt},
           "aiclk_mhz": {s: tt[s]["aiclk_mhz"] for s in tt},
           "host_load": {s: {"foreign_cores_median_max": tt[s]["foreign_cores_median"],
                             "load_per_core_median_max": tt[s]["host_load_per_core_median"]} for s in tt},
           "main_sha": {s: tt[s]["sha"] for s in tt},
           "b200_reference": {s: gpu[s] for s in gpu},
           "ratio": {s: c.get("ratio") for s, c in sizes.items()}, "verdict": verdict,
           "updated": out["written"]}
    if a.dry:
        print(json.dumps(out, indent=1, default=str))
        print(fair)
        return
    (state / "evidence" / "tt-p150a.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ev))
    (state / "VERDICT.json").write_text(json.dumps(out, indent=1) + "\n")
    (state / "FAIR.md").write_text(fair)
    (state / "rows" / f"{a.row}.json").write_text(json.dumps(row, indent=1) + "\n")
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


def fair_md(v, old_md, old_verdict):
    L = ["# RFD4 fair number: one p150a against one B200", "",
         f"Written by `perf/rfd4/fair/verdict.py` at {v['written']}. Protocol `state/rfd4/PROTOCOL.md` (frozen).",
         f"Ratio per size = {v['formula']}; per chip that is {v['per_chip_factor']}x, so {v['bar']:.2f}x needs "
         f"one p150a at >= {v['bar'] / v['per_chip_factor']:.4f}x one B200.", "",
         f"**Verdict: {v['verdict']}** against the {v['bar']:.2f}x bar. p150a firmware: {v['p150a_firmware']}.", "",
         "| size | p150a designs/s (min-max, n) | batch | AICLK MHz (per batch) | foreign cores | card, host | main sha "
         "| B200 designs/s (arm, batch, n) | ratio |", "|---|---|---|---|---|---|---|---|---|"]
    for s, c in v["sizes"].items():
        t, g = c["tt"], c["b200"]
        tc = (f"{t['designs_per_s']} ({t['min']}-{t['max']}, {t['batches']})", t["batch"],
              "/".join(str(x) for x in t["aiclk_mhz"]), t["foreign_cores_median"], f"{t['card']}, {t['host']}",
              t["sha"][:9]) if t else ("not measured", "", "", "", "", "")
        gc = (f"{g['designs_per_s']} ({g['arm']}, b{g['batch']}, {g['batches']}"
              f"{', short' if g['short'] else ''})") if g else "not published"
        L.append(f"| {s} | " + " | ".join(str(x) for x in tc) + f" | {gc} | {c.get('ratio', '')} |")
    L += ["", f"B200 side recomputed from `{v['b200_source']}`"
          + ("" if v["b200_complete"] else "; it is incomplete (missing sizes or fewer than 5 batches), "
             "so every ratio here is against what it has so far") + ".",
          f"TT measure lines rejected from the evidence: {v['tt_rejected_measure_lines'] or 'none'}."]
    return "\n".join(L) + "\n" + history(v, old_md, old_verdict)


def history(v, old_md, old_verdict):
    """The previous FAIR.md, below the new one, unless it reported the same TT measurement."""
    if not old_md.exists():
        return ""
    if old_verdict.exists():
        tt = lambda x: {str(s): c["tt"] and c["tt"]["designs_per_s"] for s, c in x["sizes"].items()}
        if tt(json.loads(old_verdict.read_text())) == tt(json.loads(json.dumps(v))):
            old = old_md.read_text()
            cut = old.find("\n## Earlier verdicts\n")
            return "" if cut < 0 else old[cut:]
    body = old_md.read_text().split("\n", 1)[1].lstrip("\n")
    head, sep, rest = body.partition("\n## Earlier verdicts\n")
    return f"\n## Earlier verdicts\n\n### Superseded at {v['written']}\n\n{head.rstrip()}\n" + (rest if sep else "")


if __name__ == "__main__":
    main()
