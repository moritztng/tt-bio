#!/usr/bin/env python3
"""Tables for state/mgx-affinity-scale.md from results/*.json and the CPU references.

    python3 perf/mgx_affinity/report.py [--refs perf/mgx_affinity/refs]

Every timed row carries its DURING-sampled AICLK median and the host load median beside it: on
whglx a wall time is capacity evidence at that load, not a speed claim.
"""
import argparse
import csv
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SCALARS = ("affinity_pred_value", "affinity_probability_binary")


def load(tag):
    f = HERE / "results" / f"{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def clk(r):
    a = (r.get("aiclk") or {})
    meds = sorted(v["median"] for v in a.values()) if a else []
    return f"{meds[len(meds) // 2]} MHz" if meds else "unclocked"


def why(r):
    for e in r.get("errors", []):
        if "allocat" in e.lower() or "memory" in e.lower():
            return e.strip()[:200]
    return (r.get("errors") or r.get("tail") or ["?"])[-1].strip()[:200]


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def size_table(prefix, grid):
    print(f"\n| {prefix} | tokens | rc | forward s | wall s | AICLK | load | first failure |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in grid:
        r = load(tag)
        if r is None:
            continue
        rows = [x for x in r["rows"] if x.get("affinity_pred_value") not in (None, "")]
        tok = rows[0].get("n_tokens", "") if rows else ""
        sec = rows[0].get("seconds") or rows[0].get("affinity_runtime_s", "") if rows else ""
        ok = r["rc"] == 0 and rows
        print(f"| {tag} | {tok} | {r['rc']} | {sec} | {r['wall_s']} | {clk(r)} | "
              f"{r['load']['median']:.0f}/{r['load']['nproc']} | {'' if ok else why(r)} |")


def ref_csv(stem, seed):
    f = HERE / "out" / f"ref_{stem}_s{seed}" / "affinity.csv"
    return {r["id"]: r for r in csv.DictReader(open(f))} if f.exists() else {}


# A screen whose failed ligands were re-run after a fix, and the tag of the re-run.
RERUN = {"b_screen_lck": "b_lck_salts"}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    tags = sorted(p.stem for p in (HERE / "results").glob("*.json"))
    size_table("nesso1", [t for t in tags if t.startswith(("n_", "nf_", "mn_")) and "screen" not in t])
    size_table("boltz2", [t for t in tags if t.startswith(("b_", "s_", "m_")) and "screen" not in t])
    for t in tags:
        r = load(t)
        if r.get("census"):
            print(f"census {t}: {r['census']}")
    kd = json.loads((HERE / "inputs/kd.json").read_text())
    for t in tags:
        if "screen" not in t:
            continue
        r = load(t)
        rows = r["rows"]
        if t in RERUN and load(RERUN[t]):  # re-run ids replace the screen's own rows
            again = {x["id"]: x for x in load(RERUN[t])["rows"]}
            rows = [again.get(x.get("id"), x) for x in rows]
        ok = [x for x in rows if x.get("affinity_pred_value") not in (None, "")]
        target = "YSK4" if "ysk4" in t else "LCK"
        k = kd[target]["records"]
        pairs = [(float(x["affinity_pred_value"]), 9 - math.log10(k[x["id"]]))
                 for x in ok if x.get("id") in k and k[x["id"]] < 10000]
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 2 else float("nan")
        errs = [(x.get("id"), x.get("error")) for x in rows if x not in ok]
        print(f"screen {t}: {len(ok)}/{len(rows)} scored, rc {r['rc']}, wall {r['wall_s']} s on "
              f"{len(r['cards'])} chip(s), {3600 * len(ok) / r['wall_s'] / len(r['cards']):.1f} "
              f"ligands/h/chip, {clk(r)}, load {r['load']['median']:.0f}; pred vs pKd Spearman "
              f"{rho:.3f} over {len(pairs)} (sign: lower pred = tighter); failed {errs}")
    print("\naccuracy nesso1 (device vs CPU fp32 reference, same seed; floor = reference seed 0 vs 1)")
    for t in tags:
        if not t.startswith(("n_", "nf_")) or "screen" in t:
            continue
        stem = load(t)["input"].split("/")[-1].removesuffix(".yaml")
        args = load(t)["args"]
        seed = args[args.index("--seed") + 1] if "--seed" in args else "default"
        a, b = ref_csv(stem, "default"), ref_csv(stem, "1")
        same = a if seed == "default" else ref_csv(stem, seed)
        dev = {x["id"]: x for x in load(t)["rows"]}
        for rid, ref in same.items():
            if rid not in dev or dev[rid].get(SCALARS[0]) in (None, ""):
                continue
            d = {s: abs(float(dev[rid][s]) - float(ref[s])) for s in SCALARS}
            f = ({s: abs(float(b[rid][s]) - float(a[rid][s])) for s in SCALARS}
                 if rid in a and rid in b else dict.fromkeys(SCALARS, float("nan")))
            print(f"  {t} ({stem}, seed {seed}): " + ", ".join(f"{s} |dev-ref| {d[s]:.4f} floor {f[s]:.4f}" for s in SCALARS))


if __name__ == "__main__":
    main()
