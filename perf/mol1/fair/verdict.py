"""Mol-1's fair ratio against the B200, from the evidence on both sides. CPU only.

    python perf/mol1/fair/verdict.py --runs <dir>/runs.jsonl [...] [--state /home/moritz/.coworker/state/mol1] [--dry]

Reads the rounds `run.py` wrote and admits, for the evidence files, only what state/mol1/PROTOCOL.md section 10
allows: warm measure rounds of `--model mol1`, not discarded (clock, contention, error, sample count), no dev
flags, from a clean tree whose sha is on tt-bio main; BH on a p150a, WH on >= 8 chips of one box. Per size and
arch it keeps, among runs with >= 5 such rounds on the newest main sha measured, the fastest, and reports its median
predictions/s with min and max. The B200 side is recomputed from `evidence/gpu-b200.jsonl` the same way the gate
does: warm lines grouped by (size, arm, batch, session), the median per group, the fastest group with >= 5 lines.

    ratio = (predictions_s_tt / chips / usd_per_tt_chip) / (predictions_s_b200 / usd_per_b200_in_a_dgx)

with prices from state/mol1/PRICES.json (PROTOCOL section 7). Writes evidence/tt-wh.jsonl, evidence/tt-p150a.jsonl,
VERDICT.json, FAIR.md (earlier verdicts kept), rows/mol1-fair-tt.json and, once every cell has both sides, the
deck's mol1.* cells.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/, for fairbox
from fairbox import carry_history  # noqa: E402

SIZES = (256, 512, 1024)
BAR, MIN_ROUNDS, MIN_WH_CHIPS = 3.00, 5, 8
ARCH = {"bh": ("p150a", "tt-p150a.jsonl", "galaxy_bh"), "wh": ("wh", "tt-wh.jsonl", "galaxy_wh")}


def jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def on_main(sha, repo):
    return bool(sha) and subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=repo,
                                        capture_output=True).returncode == 0


def admit(r, repo):
    """(arch, None) when the round is evidence, else (arch, why not)."""
    arch = {"p150a": "bh", "wh": "wh"}.get(r.get("part"))
    if r.get("kind") != "warm" or r.get("phase") != "measure":
        return arch, "not a warm measure round"
    for why in ((arch is None and f"part {r.get('part')}"), (r.get("model") != "mol1" and f"model {r.get('model')}"),
                r.get("discard"), (r.get("cli_extra") and "dev flags"), (r.get("tree_dirty") and "dirty tree"),
                (arch == "bh" and r.get("board_type") != "p150a" and f"board {r.get('board_type')}"),
                (arch == "wh" and r.get("chips", 0) < MIN_WH_CHIPS and f"{r.get('chips')} WH chips"),
                (not on_main(r.get("sha"), repo) and f"sha {str(r.get('sha'))[:9]} not on main")):
        if why:
            return arch, why
    return arch, None


def tt_side(paths, repo):
    lines, rejected = defaultdict(list), defaultdict(int)
    for p in paths:
        for r in jsonl(p):
            arch, why = admit(r, repo)
            if why is None:
                lines[arch].append(r)
            elif r.get("phase") == "measure":
                rejected[why.split(":")[0]] += 1
    per = defaultdict(dict)
    for arch, rs in lines.items():
        for size in SIZES:
            runs = defaultdict(list)
            for r in rs:
                if r["size"] == size:
                    runs[(r["raw"], r["sha"], r["batch"], r["chips"])].append(r)
            full = [v for v in runs.values() if len(v) >= MIN_ROUNDS]
            if not full:
                continue
            sha = max(full, key=lambda v: max(r["t"] for r in v))[0]["sha"]
            v = max((v for v in full if v[0]["sha"] == sha),
                    key=lambda v: statistics.median(r["predictions_per_s"] for r in v))
            d = [r["predictions_per_s"] for r in v]
            per[arch][size] = {
                "predictions_per_s": round(statistics.median(d), 6), "min": round(min(d), 6),
                "max": round(max(d), 6), "rounds": len(v), "batch": v[0]["batch"], "chips": v[0]["chips"],
                "chip_ids": v[0]["chip_ids"], "sha": sha, "host": v[0]["host"], "board_type": v[0].get("board_type"),
                "firmware": v[0].get("firmware"), "arm": v[0].get("arm"), "raw": v[0]["raw"],
                "aiclk_mhz": [r["aiclk_mhz"] for r in v], "aiclk_min_mhz": min(r["aiclk_min_mhz"] for r in v),
                "foreign_cores_median": max(r.get("foreign_cores_median") or 0 for r in v),
                "host_load_per_core_median": max(r.get("host_load_per_core_median") or 0 for r in v),
                "date": max(r["t"] for r in v), "lines": v}
    return per, dict(rejected)


def gpu_side(path):
    groups = defaultdict(list)
    for r in jsonl(path):
        if r.get("kind") == "warm" and (r.get("wall_s") or 0) > 0 and (r.get("predictions") or 0) > 0:
            groups[(r["size"], r.get("arm"), r.get("batch"), r.get("session"))].append(r)
    per = {}
    for (size, arm, b, session), v in groups.items():
        d = [r["predictions"] / r["wall_s"] for r in v]
        cand = {"predictions_per_s": round(statistics.median(d), 6), "min": round(min(d), 6),
                "max": round(max(d), 6), "arm": arm, "batch": b, "session": session, "batches": len(v),
                "gpu": v[0].get("gpu"), "host": v[0].get("host"), "sha": v[0].get("sha"),
                "clock_mhz": statistics.median(r.get("clock_mhz") or 0 for r in v),
                "date": max(r.get("t", "") for r in v), "short": len(v) < MIN_ROUNDS}
        cur = per.get(size)
        if cur is None or (cur["short"], -cur["predictions_per_s"]) > (cand["short"], -cand["predictions_per_s"]):
            per[size] = cand
    return per


def evidence_line(r):
    keep = ("kind", "size", "tokens", "predictions", "wall_s", "predictions_per_s", "batch", "chips", "chip_ids",
            "part", "board_type", "firmware", "host", "arm", "sha", "seed", "aiclk_mhz", "aiclk_per_chip",
            "aiclk_min_mhz", "aiclk_samples", "foreign_cores_median", "foreign_device_holders",
            "host_load_per_core_median", "work", "raw", "t")
    return {k: r.get(k) for k in keep} | {"coords_sha256": [c["coords_sha256"] for c in r["per_chip"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--state", default="/home/moritz/.coworker/state/mol1")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    state = Path(a.state)
    prices = json.loads((state / "PRICES.json").read_text())
    usd = {k: prices[k]["usd"] / prices[k].get("chips", prices[k].get("gpus")) for k in ("galaxy_wh", "galaxy_bh", "dgx_b200")}

    tt, rejected = tt_side(a.runs, a.repo)
    gpu = gpu_side(state / "evidence" / "gpu-b200.jsonl")
    sizes = {}
    for s in SIZES:
        g = gpu.get(s)
        cell = {"b200": g and {k: g[k] for k in ("arm", "batch", "session", "gpu", "predictions_per_s", "min", "max",
                                                  "batches", "clock_mhz", "short")}}
        for arch, (_, _, price) in ARCH.items():
            t = tt.get(arch, {}).get(s)
            if not t:
                continue
            cell[arch] = {k: v for k, v in t.items() if k != "lines"}
            if g and not g["short"]:
                cell[arch]["ratio"] = round((t["predictions_per_s"] / t["chips"] / usd[price])
                                            / (g["predictions_per_s"] / usd["dgx_b200"]), 3)
        sizes[str(s)] = cell
    ratios = [sizes[str(s)].get(arch, {}).get("ratio") for s in SIZES for arch in ARCH]
    verdict = ("PASS" if min(ratios) >= BAR else "FAIL") if all(r is not None for r in ratios) else "INCOMPLETE"
    written = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = {"verdict": verdict, "bar": BAR, "sizes": sizes, "usd_per_chip": usd, "tt_rejected_measure_rounds": rejected,
           "formula": "(predictions_s_tt / chips / usd_per_tt_chip) / (predictions_s_b200 / usd_per_b200_in_dgx)",
           "protocol": "state/mol1/PROTOCOL.md (sections 1-9 GPU and work, 10 TT)", "written": written}
    row = {"status": "done" if verdict in ("PASS", "FAIL") else "running",
           "platforms": {a_: ARCH[a_][0] for a_ in tt}, "hosts": sorted({t["host"] for c in tt.values() for t in c.values()}),
           "cards": {a_: sorted({i for t in c.values() for i in t["chip_ids"]}) for a_, c in tt.items()},
           "fw": {a_: sorted({str(t["firmware"]) for t in c.values()}) for a_, c in tt.items()}, "sizes": list(SIZES),
           "rate_per_size": {a_: {s: t["predictions_per_s"] for s, t in c.items()} for a_, c in tt.items()},
           "batch": {a_: {s: t["batch"] for s, t in c.items()} for a_, c in tt.items()},
           "aiclk_mhz": {a_: {s: t["aiclk_mhz"] for s, t in c.items()} for a_, c in tt.items()},
           "host_load": {a_: {s: {"foreign_cores_median_max": t["foreign_cores_median"],
                                  "load_per_core_median_max": t["host_load_per_core_median"]} for s, t in c.items()}
                         for a_, c in tt.items()},
           "main_sha": {a_: {s: t["sha"] for s, t in c.items()} for a_, c in tt.items()},
           "b200_reference": {s: c["b200"] for s, c in sizes.items()},
           "ratio": {s: {a_: c.get(a_, {}).get("ratio") for a_ in ARCH} for s, c in sizes.items()},
           "verdict": verdict, "updated": written}
    fair = fair_md(out)
    if a.dry:
        print(json.dumps(out, indent=1, default=str))
        print(fair)
        return
    for arch, (_, ev, _) in ARCH.items():
        lines = [evidence_line(r) for t in tt.get(arch, {}).values() for r in t["lines"]]
        (state / "evidence" / ev).write_text("".join(json.dumps(x) + "\n" for x in lines))
    (state / "VERDICT.json").write_text(json.dumps(out, indent=1) + "\n")
    (state / "FAIR.md").write_text(carry_history(state / "FAIR.md", fair, written))
    (state / "rows" / "mol1-fair-tt.json").write_text(json.dumps(row, indent=1) + "\n")
    if verdict in ("PASS", "FAIL"):
        deck = state.parent / "biodeck" / "src" / "extra_cells.json"
        cells = {k: v for k, v in json.loads(deck.read_text()).items() if not k.startswith("mol1.")}
        cells.update(deck_cells(out))
        deck.write_text(json.dumps(cells, indent=1) + "\n")
    print(fair)


def deck_cells(v):
    """mol1.<side>.<size>: one measured cell per size and side, never a projection."""
    cells = {}
    for s, c in v["sizes"].items():
        for arch in ARCH:
            t = c[arch]
            cells[f"mol1.{ARCH[arch][0]}.{s}"] = {
                "s": round(t["chips"] / t["predictions_per_s"], 3),
                "unit": f"s/prediction per chip (warm median of {t['rounds']} rounds on {t['chips']} chip(s), "
                        f"{t['min']}-{t['max']} predictions/s)",
                "n": t["rounds"], "aiclk_mhz": statistics.median(t["aiclk_mhz"]), "batch": t["batch"],
                "chips": t["chips"], "versions": f"tt-bio main {t['sha'][:9]}, Mol-1 mol-1-base, 20 recycles / 200 "
                f"steps / 5 samples, mol1-{s} frozen input, firmware {t['firmware']}, arm {t['arm']}",
                "when": t["date"], "card": f"{t['host']}:{t['chip_ids']} {t['board_type'] or 'Wormhole Galaxy'}",
                "source": f"{t['raw']}/runs.jsonl (perf/mol1/fair/run.py)", "ratio_vs_b200": t["ratio"]}
        g = c["b200"]
        cells[f"mol1.b200.{s}"] = {
            "s": round(1 / g["predictions_per_s"], 3), "n": g["batches"], "clock_mhz": g["clock_mhz"],
            "unit": f"s/prediction (warm median of {g['batches']} batches at K={g['batch']}, arm {g['arm']})",
            "batch": g["batch"], "versions": f"Mol-1 upstream e6e64866, arm {g['arm']}, mol1-{s} frozen input",
            "box": f"1x {g['gpu']} (session {g['session']})", "source": "state/mol1/evidence/gpu-b200.jsonl"}
    return cells


def fair_md(v):
    L = ["# MOL1 fair number: Wormhole Galaxy and p150a against one B200", "",
         f"Written by `perf/mol1/fair/verdict.py` at {v['written']}. Protocol {v['protocol']}.",
         f"Ratio per size = {v['formula']}; $ per chip: WH {v['usd_per_chip']['galaxy_wh']:.2f}, "
         f"BH {v['usd_per_chip']['galaxy_bh']:.2f}, B200 in a DGX {v['usd_per_chip']['dgx_b200']:.2f}.", "",
         f"**Verdict: {v['verdict']}** against the {v['bar']:.2f}x bar.", "",
         "| size | side | predictions/s (min-max, n) | chips x batch | AICLK MHz (per round) | foreign cores | host | "
         "main sha | fw | ratio |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, c in v["sizes"].items():
        for arch in ARCH:
            t = c.get(arch)
            L.append(f"| {s} | {arch} | " + (" | ".join(str(x) for x in (
                f"{t['predictions_per_s']} ({t['min']}-{t['max']}, {t['rounds']})", f"{t['chips']} x {t['batch']}",
                "/".join(str(x) for x in t["aiclk_mhz"]), t["foreign_cores_median"], t["host"], t["sha"][:9],
                t["firmware"], t.get("ratio", ""))) if t else "not measured | | | | | | |") + " |")
        g = c["b200"]
        L.append(f"| {s} | b200 | " + (f"{g['predictions_per_s']} ({g['min']}-{g['max']}, {g['batches']}"
                                       f"{', short' if g['short'] else ''}) | K={g['batch']}, {g['arm']} | "
                                       f"SM {g['clock_mhz']} | | {g['session']} | | | |" if g else
                                       "not published | | | | | | | |"))
    L += ["", f"TT measure rounds rejected: {v['tt_rejected_measure_rounds'] or 'none'}.",
          "Chassis: the B200 side is priced as a DGX B200; the HGX exposure (state/rfd4/GPU-AUDIT.md section 5) "
          "applies unchanged and is quoted beside every published ratio."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
