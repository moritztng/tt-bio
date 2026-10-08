"""Summarize bench.py records into BOARD lines, optionally against a base arm.

    python perf/spd/summarize.py RUN/*/bench.jsonl [--base exact] [--row spd-trimul] [--mode normal]

Per (arch, chip, arm, input): warm reps only (cold is reported apart), each finite and error-free; mean, spread
(max - min), AICLK median and lowest sample over the reps. With --base, x = base mean / arm mean on the same
chip, and the structural deviation from the base arm's same-seed fold: all-atom RMSD after Kabsch fit, per
diffusion sample, reported as the max over samples and seeds, in Angstrom (the charter's kill bar is 0.60 A at
512 aa; re-running exact with another seed moves a 512 aa structure 1.84 A).

A row is marked CLOCK when its AICLK is not trustworthy: any sample below 95 % of its median, or a Blackhole
median under 1200 MHz. Such a number is an artifact, not a regression.
"""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("records", nargs="+", type=Path)
ap.add_argument("--base")
ap.add_argument("--row", default="?")
ap.add_argument("--mode", default="normal", choices=["normal", "fast"])
a = ap.parse_args()

reps = []
for f in a.records:
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("ev") == "rep":
            r["_dir"] = f.parent
            reps.append(r)

groups = defaultdict(list)
for r in reps:
    groups[(r["arch"], r["host"], r["chip"], r["arm"], r["input"])].append(r)


def clk(rs):
    med = [n["median"] for r in rs for n in r["aiclk"].values()]
    lo = [n["min"] for r in rs for n in r["aiclk"].values()]
    return (int(statistics.median(med)) if med else None), (min(lo) if lo else None)


def rmsd_vs(r, base):
    """Max over samples of all-atom Kabsch RMSD between two same-seed folds, Angstrom."""
    import torch
    fa, fb = (x["_dir"] / f"coords_{x['input']}_s{x['seed']}.pt" for x in (r, base))
    if not (fa.exists() and fb.exists()):
        return None
    ca, cb = torch.load(fa)["coords"].double(), torch.load(fb)["coords"].double()
    ca, cb = ca.reshape(-1, *ca.shape[-2:]), cb.reshape(-1, *cb.shape[-2:])
    if ca.shape != cb.shape:
        return None
    worst = 0.0
    for p, q in zip(ca, cb):
        keep = (p.abs().sum(-1) > 0) & (q.abs().sum(-1) > 0)
        p, q = p[keep] - p[keep].mean(0), q[keep] - q[keep].mean(0)
        u, _s, vt = torch.linalg.svd(p.T @ q)
        d = torch.sign(torch.linalg.det(u @ vt))
        rot = u @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=p.dtype)) @ vt
        worst = max(worst, float(((p @ rot - q) ** 2).sum(-1).mean().sqrt()))
    return worst


out = []
for key in sorted(groups):
    arch, host, chip, arm, inp = key
    rs = groups[key]
    warm = [r for r in rs if r["kind"] == "warm" and r["finite"] and not r["err"] and r["fold_s"]]
    bad = [r for r in rs if not r["finite"] or r["err"]]
    cold = [r["fold_s"] for r in rs if r["kind"] == "cold" and r["fold_s"]]
    if not warm:
        out.append(f"{arch} {host} chip {chip} {arm} {inp}: no clean warm rep ({len(bad)} failed)")
        continue
    t = [r["fold_s"] for r in warm]
    mean, spread = statistics.mean(t), max(t) - min(t)
    med, lo = clk(warm)
    flag = []
    if med and lo and lo < 0.95 * med:
        flag.append("CLOCK")
    if arch.startswith("blackhole") and med and med < 1200:
        flag.append("CLOCK")
    x, dev = "", ""
    if a.base and arm != a.base:
        b = groups.get((arch, host, chip, a.base, inp))
        bw = [r for r in b or [] if r["kind"] == "warm" and r["finite"] and not r["err"] and r["fold_s"]]
        if bw:
            x = f"{statistics.mean(r['fold_s'] for r in bw) / mean:.3f}x vs {a.base}"
            by_seed = {r["seed"]: r for r in b}
            ds = [d for r in rs if r["seed"] in by_seed for d in [rmsd_vs(r, by_seed[r["seed"]])] if d is not None]
            if ds:
                dg = sum(r["digest"] == by_seed[r["seed"]]["digest"] for r in rs if r["seed"] in by_seed)
                dev = f"max dev {max(ds):.3f} A over {len(ds)} seeds ({dg} bit-identical)"
    sha = (warm[0]["sha"] or "?")[:9] + ("+dirty" if warm[0]["dirty"] else "")
    out.append(" | ".join([
        "", a.row, "protenix-v2", arch, f"{host}/c{chip}", a.mode, inp, f"{warm[0].get('tokens')}", arm, sha,
        f"{mean:.2f} s (n={len(t)}, spread {spread:.2f}{', cold ' + format(cold[0], '.1f') if cold else ''})",
        f"{med}/{lo} MHz{' ' + ','.join(sorted(set(flag))) if flag else ''}", x or "-", dev or "-",
        f"{len(bad)} failed" if bad else "finite", str(warm[0]["_dir"]), ""]).strip())
print("\n".join(out))
