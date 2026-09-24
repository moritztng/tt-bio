#!/usr/bin/env python3
"""Speed-bar verdicts for the designers, embedding models and affinity surfaces, from runs/<model>.jsonl.

scripts/speed_bar.judge is called unchanged, with all three guards on. Per rung: runtime = median of
its timed units (warm-ups are never timed), AICLK = median of the units' DURING medians, load = the
worst 1-min loadavg/nproc any unit saw, identity = (host, chip, engine tree, host thread cap), one
value per rung. When a rung has units under the load ceiling only those are judged. The fit takes the
rungs in 512..1024 whose AICLK is within the bar's CLOCK_TOL of the judged rung's, and sigma is the
relative stdev at the fit rung timed most often (512 unless a model's 512 is off-clock). That is
mgx-speed's route (a) for nesso1, whose small rungs are host-bound at 500 MHz; where every rung
reads one clock it changes nothing. order = 3 for pair models (designers, affinity), 2 for the
sequence-only embedders, as the brief and docs/speed-bar.md set it.

Throughput per rung is what a sponsor sizes against: designs/h/chip = 3600 / (s per design),
sequences/s/chip = 1 / (s per single-sequence call), affinity jobs/h/chip = 3600 / runtime_s.

    python perf/mgx_speed_a4/verdicts.py [runs_dir]     # one JSON object per model, then a table
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import speed_bar as sb  # noqa: E402

SIGMA_RUNG = 512
ORDER = {"design": 3, "affinity": 3, "embed": 2}
UNIT = {"design": ("designs/h/chip", lambda t: 3600 / t), "embed": ("seq/s/chip", lambda t: 1 / t),
        "affinity": ("jobs/h/chip", lambda t: 3600 / t)}


def model_verdicts(lines: list[dict]) -> dict:
    family = lines[0]["family"]
    quiet = lambda c: (c.get("load") or {}).get("max", sb.DEFAULT_LOAD_CEILING + 1) <= sb.DEFAULT_LOAD_CEILING
    has_quiet = {c["rung"] for c in lines if c["tag"] != "warmup" and not c.get("error") and quiet(c)}
    timed, failed = defaultdict(list), {}
    ident, clk, load = defaultdict(set), defaultdict(list), defaultdict(list)
    for c in lines:
        n = c["rung"]
        if c.get("contended"):
            continue
        if c.get("error"):
            failed[n] = c["error"]
            continue
        if c["tag"] == "warmup" or (n in has_quiet and not quiet(c)):
            continue
        timed[n].append(c["runtime_s"])
        ident[n].add((c["host"], c["card"], c["engine"], c["host_threads"]))
        if c.get("aiclk"):
            clk[n].append(c["aiclk"]["median"])
        load[n].append((c.get("load") or {}).get("max", float("inf")))
    rt = {n: statistics.median(v) for n, v in timed.items()}
    aiclk = {n: statistics.median(v) for n, v in clk.items()}
    unit, rate = UNIT[family]
    out = {"family": family, "order": ORDER[family], "runtimes": rt, "aiclk": aiclk,
           "load_max": {n: max(v) for n, v in load.items()}, "coverage": failed,
           "throughput": {n: round(rate(t), 3) for n, t in rt.items()}, "throughput_unit": unit,
           "rungs": {}}
    for n in sorted(r for r in {*rt, *failed} if r > sb.FIT_HI):
        if n in failed and n not in rt:
            out["rungs"][n] = {"verdict": "COVERAGE", "why": failed[n][:300]}
            continue
        if n not in aiclk:
            out["rungs"][n] = {"verdict": "VOID", "why": "the rung has no DURING-sampled AICLK"}
            continue
        fit_rungs = sorted(r for r in rt if sb.FIT_LO <= r <= sb.FIT_HI and r in aiclk
                           and abs(aiclk[r] - aiclk[n]) / aiclk[n] <= sb.CLOCK_TOL)
        sr = max(fit_rungs, key=lambda r: (len(timed[r]), r == SIGMA_RUNG), default=None)
        s = timed[sr] if sr else []
        if len(s) < 2:
            out["rungs"][n] = {"verdict": "VOID", "why": "no clock-matched fit rung has 2 timed units"}
            continue
        sigma = statistics.stdev(s) / statistics.mean(s)
        rungs = [*fit_rungs, n]
        if any(len(ident[r]) != 1 for r in rungs):
            out["rungs"][n] = {"verdict": "VOID", "why": "units within one rung differ in identity"}
            continue
        v = sb.judge({r: rt[r] for r in fit_rungs}, n, rt[n], order=ORDER[family], sigma=sigma,
                     aiclk={r: aiclk[r] for r in rungs},
                     identity={r: next(iter(ident[r])) for r in rungs},
                     load={r: out["load_max"][r] for r in rungs})
        out["rungs"][n] = {**v, "fit_rungs": fit_rungs, "sigma": round(sigma, 4), "sigma_rung": sr,
                           "sigma_reps": s, "runtime_s": rt[n], "aiclk": aiclk[n],
                           "load": out["load_max"][n]}
    return out


def main(argv):
    runs = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent / "runs"
    table = []
    for f in sorted(runs.glob("*.jsonl")):
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if not lines:
            continue
        v = {"model": f.stem, **model_verdicts(lines)}
        print(json.dumps(v, default=str))
        for n, r in v["rungs"].items():
            table.append(f"{f.stem:16s} {n:5d} {r['verdict']:8s} "
                         + (f"t={r['runtime_s']:.4g} pred={r['predicted_s']} ratio={r['ratio']} "
                            f"allowed={r['allowed']} k_fit={r['k_fit']} sigma={r['sigma']:.2%}@{r['sigma_rung']} "
                            f"clk={r['aiclk']} load={r['load']:.2f}" if "ratio" in r else r["why"]))
    print("\n".join(table))


if __name__ == "__main__":
    main(sys.argv)
