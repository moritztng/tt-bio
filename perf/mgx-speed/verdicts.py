"""Speed-bar verdict for every timed rung above 1024, from perf/mgx-speed/runs/<model>.jsonl.

Per rung: runtime = median of its timed folds (warm-ups are not timed), AICLK = median of the
folds' DURING medians, load = the worst 1-min loadavg/nproc any of its folds saw, identity =
(host, chip, commit, host thread cap), which must be one value per rung. sigma is the relative
stdev at the ladder's sigma rung (512). Then scripts/speed_bar.judge with all three guards on.
A rung whose folds refused or crashed is reported as a coverage result and never judged.

    python perf/mgx-speed/verdicts.py [runs_dir]      # prints one JSON object per model
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import speed_bar as sb  # noqa: E402

SIGMA_RUNG, ORDER = 512, 3          # every model timed here has a pair track (triangle ops)


def model_verdicts(lines: list[dict]) -> dict:
    timed, failed, ident, clk, load = defaultdict(list), {}, defaultdict(set), defaultdict(list), defaultdict(list)
    for c in lines:
        n = c["rung"]
        if c.get("error") or c.get("refused"):
            failed[n] = c.get("refused") or c["error"]
            continue
        if c["tag"] == "warmup":
            continue
        timed[n].append(c["runtime_s"])
        ident[n].add((c["host"], c["card"], c["commit"], c["host_threads"]))
        if c.get("aiclk"):
            clk[n].append(c["aiclk"]["median"])
        load[n].append(c["load"]["max"])
    rt = {n: statistics.median(ts) for n, ts in timed.items()}
    s = timed.get(SIGMA_RUNG, [])
    sigma = statistics.stdev(s) / statistics.mean(s) if len(s) > 1 else 0.0
    out = {"sigma": round(sigma, 4), "sigma_reps": s, "runtimes": rt,
           "aiclk": {n: statistics.median(v) for n, v in clk.items()},
           "load_max": {n: max(v) for n, v in load.items()}, "coverage": failed, "rungs": {}}
    fit_rungs = [n for n in rt if sb.FIT_LO <= n <= sb.FIT_HI]
    for n in sorted(r for r in {*rt, *failed} if r > sb.FIT_HI):
        if n in failed:
            out["rungs"][n] = {"verdict": "COVERAGE", "why": failed[n][:300]}
            continue
        rungs = [*fit_rungs, n]
        if any(len(ident[r]) != 1 for r in rungs):
            out["rungs"][n] = {"verdict": "VOID", "why": "folds within one rung differ in identity"}
            continue
        if any(r not in out["aiclk"] for r in rungs):
            out["rungs"][n] = {"verdict": "VOID", "why": "a rung has no DURING-sampled AICLK"}
            continue
        v = sb.judge({r: rt[r] for r in fit_rungs}, n, rt[n], order=ORDER, sigma=sigma,
                     aiclk={r: out["aiclk"][r] for r in rungs},
                     identity={r: next(iter(ident[r])) for r in rungs},
                     load={r: out["load_max"][r] for r in rungs})
        out["rungs"][n] = {**v, "runtime_s": rt[n], "aiclk": out["aiclk"][n], "load": out["load_max"][n]}
    return out


def main(argv):
    runs = Path(argv[1]) if len(argv) > 1 else ROOT / "perf" / "mgx-speed" / "runs"
    for f in sorted(runs.glob("*.jsonl")):
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        print(json.dumps({"model": f.stem, **model_verdicts(lines)}, default=str))


if __name__ == "__main__":
    main(sys.argv)
