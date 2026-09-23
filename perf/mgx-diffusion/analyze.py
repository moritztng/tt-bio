"""Read perf/mgx-diffusion/runs.jsonl into the row's LADDER / SCALING / CEILING / RECYCLES tables.

    python perf/mgx-diffusion/analyze.py [runs.jsonl] [engine tree sha, default the last row's]

Drops folds that never ran (the chip was taken between folds) and, for timing, folds that
compiled (cold) or ran with the DRAM probe on (the probe drains the pipeline at every tag).
The sample law is a least-squares fit of runtime_s = a + b * samples per (model, tokens) over
production-step folds; `b/a` is what one more sample costs as a fraction of a 1-sample fold.
Memory is the probe's peak over the whole fold and over the diffusion + confidence tags alone.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("runs.jsonl")
rows = [json.loads(x) for x in RUNS.read_text().splitlines() if x.strip()]
rows = [r for r in rows if "is in use by" not in (r.get("error") or "")]
# One engine tree at a time: a point folded on another engine is a different measurement.
ENGINE = sys.argv[2] if len(sys.argv) > 2 else rows[-1]["engine"]
rows = [r for r in rows if r["engine"] == ENGINE]


def clk(r):
    a = r.get("aiclk") or {}
    return f"{a.get('median', '?')}" if a else "?"


def fail(r):
    return r.get("error") or r.get("refused")


def fit(pts):
    """runtime = a + b*s by least squares; None under two distinct sample counts."""
    xs = [s for s, _t in pts]
    if len(set(xs)) < 2:
        return None
    n, mx = len(pts), sum(xs) / len(pts)
    my = sum(t for _s, t in pts) / n
    sxx = sum((s - mx) ** 2 for s in xs)
    b = sum((s - mx) * (t - my) for s, t in pts) / sxx
    a = my - b * mx
    worst = max(abs(t - (a + b * s)) / t for s, t in pts)
    return a, b, worst


def main():
    base = [r for r in rows if r.get("recycles") is None]
    print(f"engine tree {ENGINE[:12]}, commits {sorted({r['commit'][:9] for r in rows})}")
    print("LADDER (samples x tokens; time = production steps, warm, probe off; memory = 6-step probe folds)")
    cell = defaultdict(dict)
    for r in base:
        k = (r["model"], r["tokens"], r["samples"])
        if r.get("probe") and r.get("steps") == 6:
            cell[k]["mem"] = r
        elif not r.get("probe") and r.get("steps") is None and (not r.get("cold") or fail(r)):
            cell[k]["time"] = r
    for (m, t, s) in sorted(cell, key=lambda k: (k[0], k[1], k[2])):
        c = cell[(m, t, s)]
        mem, tim = c.get("mem"), c.get("time")
        ms = "-"
        if mem is not None:
            if fail(mem):
                ms = f"FAIL({fail(mem)[:80]})"
            elif mem.get("dram"):
                d = mem["dram"]
                sp = (d.get("sample_phase") or {}).get("gib")
                ms = f"peak {d['peak_gib']:.2f} GiB @ {d['peak_tag'][:40]!r}; sample-phase {sp if sp is None else round(sp, 2)} GiB"
            if mem.get("n_struct") != s and not fail(mem):
                ms += f" [n_struct {mem.get('n_struct')}]"
        ts = "-"
        if tim is not None:
            if fail(tim):
                ts = f"FAIL({fail(tim)[:80]})"
            else:
                ph = (tim.get("progress") or {}).get("phase_s") or {}
                ts = (f"{tim['runtime_s']:.1f} s (trunk {ph.get('trunk', '?')} s, diffusion "
                      f"{ph.get('diffusion', '?')} s, confidence {ph.get('confidence', '?')} s) "
                      f"AICLK {clk(tim)} load {(tim.get('load') or {}).get('max', '?')}")
        print(f"  {m:14s} {t:5d} x {s:3d}: time {ts} | mem {ms}")

    print("\nSCALING (runtime_s = a + b*samples, production steps)")
    by = defaultdict(list)
    for (m, t, s), c in cell.items():
        tim = c.get("time")
        if tim is not None and not fail(tim) and tim.get("runtime_s"):
            by[(m, t)].append((s, tim["runtime_s"]))
    for (m, t), pts in sorted(by.items()):
        f = fit(pts)
        if f:
            a, b, worst = f
            print(f"  {m:14s} {t:5d}: a={a:7.1f} s  b={b:6.2f} s/sample  b/a={b / a:5.3f}  "
                  f"worst residual {100 * worst:.1f}%  (n={len(pts)})")
    mem_by = defaultdict(list)
    for (m, t, s), c in cell.items():
        mem = c.get("mem")
        if mem is not None and not fail(mem) and mem.get("dram"):
            mem_by[(m, t)].append((s, mem["dram"]["peak_gib"],
                                   (mem["dram"].get("sample_phase") or {}).get("gib")))
    for (m, t), pts in sorted(mem_by.items()):
        pts.sort()
        print(f"  mem {m:14s} {t:5d}: " + ", ".join(
            f"S={s}: {p:.2f}" + (f" ({sp:.2f})" if sp else "") for s, p, sp in pts))

    print("\nRECYCLES (cycles run per request, progress stream)")
    for r in sorted((r for r in rows if r.get("recycles") is not None),
                    key=lambda r: (r["model"], r["tokens"], r["recycles"])):
        p = r.get("progress") or {}
        d = (r.get("dram") or {}).get("peak_gib")
        print(f"  {r['model']:14s} {r['tokens']:5d} R={r['recycles']:2d}: trunk total "
              f"{p.get('trunk_total')} ({p.get('trunk_steps_seen')} seen), runtime "
              f"{r.get('runtime_s')} s, trunk {((p.get('phase_s') or {}).get('trunk'))} s"
              + (f", peak {d:.2f} GiB" if d else "") + (f" FAIL {fail(r)[:80]}" if fail(r) else "")
              + (" cold" if r.get("cold") else ""))


if __name__ == "__main__":
    main()
