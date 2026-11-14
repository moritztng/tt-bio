"""Summarize bench.py records into BOARD lines, optionally against a base arm.

    python perf/spd/summarize.py RUN/*/bench.jsonl [--base exact] [--row spd-trimul] [--mode normal]

Per (arch, chip, arm, input): warm reps only (cold is reported apart), each finite and error-free; mean, spread
(max - min), AICLK median and lowest sample over the reps. With --base, x = base mean / arm mean on the same
chip, and the structural deviation from the base arm's same-seed fold: all-atom RMSD after Kabsch fit, per
diffusion sample, reported as the max over samples and seeds, in Angstrom (the charter's kill bar is 0.60 A at
512 aa; re-running exact with another seed moves a 512 aa structure 1.84 A).

A row is marked CLOCK when its AICLK is not trustworthy: any sample below 95 % of its median, or a Blackhole
median under 1200 MHz. Such a number is an artifact, not a regression. LOAD: the host's 1-minute load average
exceeded its CPU count during a warm rep, so the host part of the fold was contended. VOID: a fast arm whose
engine built exact (bench.py before 530a4520 never turned fast mode on); never post it as a fast number.
"""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("records", nargs="+", type=Path)
ap.add_argument("--base")
ap.add_argument("--row", default="?")
ap.add_argument("--mode", default="normal", choices=["normal", "fast"])
ap.add_argument("--data", type=Path, default=Path.home() / "spd-data", help="for MANIFEST.tsv token counts")
a = ap.parse_args()
# Boltz-2's metrics carry no token count; the input's manifest line does.
MANIFEST = {}
if (a.data / "MANIFEST.tsv").exists():
    for line in (a.data / "MANIFEST.tsv").read_text().splitlines():
        name, *kv = line.split("\t")
        MANIFEST[name] = dict(x.split("=", 1) for x in kv if "=" in x).get("tokens")

reps = []
for f in a.records:
    built_fast, model = None, "protenix-v2"
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("ev") == "start":
            model = r.get("model") or model
        if r.get("ev") == "build":
            # A Protenix-family model built through the worker captures fast mode and resets the global,
            # so the build record reads False on a real fast build (bench.py records the model's own flag
            # from 2026-10-09 on). Only Protenix-v2 builds by hand and can fold exact under `:fast`.
            built_fast = r.get("fast") or model not in ("protenix-v2",)
        if r.get("ev") == "rep":
            r["_dir"], r["_built_fast"], r["_model"] = f.parent, built_fast, model
            reps.append(r)

groups = defaultdict(list)
for r in reps:
    groups[(r["_model"], r["arch"], r["host"], r["chip"], r["arm"], r["input"])].append(r)


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
    model, arch, host, chip, arm, inp = key
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
    if any(r.get("ncpu") and max(r["loadavg"]) > r["ncpu"] for r in warm):
        flag.append("LOAD")
    if any(r.get("fast") and not r["_built_fast"] for r in warm):
        flag.append("VOID")
    x, dev = "", ""
    if a.base and arm != a.base:
        b = groups.get((model, arch, host, chip, a.base, inp))
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
        "", a.row, warm[0]["_model"], arch, f"{host}/c{chip}", a.mode, inp, f"{warm[0].get('tokens') or MANIFEST.get(inp)}",
        arm, sha,
        f"{mean:.2f} s (n={len(t)}, spread {spread:.2f}{', cold ' + format(cold[0], '.1f') if cold else ''})",
        f"{med}/{lo} MHz{' ' + ','.join(sorted(set(flag))) if flag else ''}", x or "-", dev or "-",
        f"{len(bad)} failed" if bad else "finite", str(warm[0]["_dir"]), ""]).strip())
print("\n".join(out))
