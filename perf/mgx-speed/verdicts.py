"""Speed-bar verdict for every timed rung above 1024, from perf/mgx-speed/runs/<model>.jsonl.

Per rung: runtime = median of its timed folds (warm-ups are not timed), AICLK = median of the
folds' DURING medians, load = the worst 1-min loadavg/nproc any of its folds saw, identity =
(host, chip, engine tree, host thread cap), which must be one value per rung. The engine tree is
the commit's tt_bio/ and scripts/ trees, so a tooling commit under perf/ does not split a model's
rungs. A fold that saw load above the ceiling is VOID and was re-run (time_rungs.py): when a rung
has quiet folds only those are judged, and a rung with none stays VOID. sigma is the relative
stdev at the ladder's sigma rung (512). Then scripts/speed_bar.judge with all three guards on.
A rung whose folds refused or crashed is reported as a coverage result and never judged.

Two orchestrator rulings (2026-09-23 14:23Z) sit on top of the bar without editing it:
NOT QUOTABLE replaces a PASS/FAIL whose fit sigma is more than 3x the model's quiet fit sigma
(QUIET_SIGMA; the fit is re-taken, the bar's verdict is kept beside it). nesso1's 512-768 rungs
are host-bound and idle at 500 MHz, so it fits only the rungs whose clock matches the judged
rung's and takes its sigma inside that fit (route (a)); fewer than three is UNGATED.

    python perf/mgx-speed/verdicts.py [runs_dir]      # prints one JSON object per model
"""
import functools
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import speed_bar as sb  # noqa: E402

SIGMA_RUNG, ORDER = 512, 3          # every model timed here has a pair track (triangle ops)
MODEL_SIGMA_RUNG = {"nesso1": 960}
CLOCK_MATCHED_FIT = {"nesso1"}
# The sigma rung's relative stdev from the quiet window (1-min load <= 0.65x nproc): C3 f16b98e5f
# 08:40-09:29Z, esmfold2-fast on dcbe5d805. A model absent here never had a quiet sigma rung.
QUIET_SIGMA = {"openbind": 0.0303, "openfold3": 0.0569, "opendde": 0.0097, "opendde-abag": 0.0058,
               "protenix-v2": 0.0029, "rf3": 0.0155, "esmfold2-fast": 0.0072}


@functools.lru_cache(None)
def engine(commit: str) -> str:
    git = subprocess.run(["git", "rev-parse", f"{commit}:tt_bio", f"{commit}:scripts"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return git.stdout.strip().replace("\n", " ")


def quiet_only(lines: list[dict]) -> list[dict]:
    """Drop a rung's load-voided timed folds when the rung also has quiet ones."""
    quiet = lambda c: c.get("load", {}).get("max", sb.DEFAULT_LOAD_CEILING + 1) <= sb.DEFAULT_LOAD_CEILING
    has_quiet = {c["rung"] for c in lines if c["tag"] != "warmup" and quiet(c)}
    return [c for c in lines if c["tag"] == "warmup" or c["rung"] not in has_quiet or quiet(c)
            or c.get("error") or c.get("refused")]


def model_verdicts(model: str, lines: list[dict]) -> dict:
    timed, failed, ident, clk, load = defaultdict(list), {}, defaultdict(set), defaultdict(list), defaultdict(list)
    for c in quiet_only(lines):
        n = c["rung"]
        if (c.get("error") or "").startswith("census fold exited 75"):
            continue                # the lease refused a chip another row held: nothing ran
        if c.get("error") or c.get("refused"):
            failed[n] = c.get("refused") or c["error"]
            continue
        if c["tag"] == "warmup":
            continue
        timed[n].append(c["runtime_s"])
        ident[n].add((c["host"], c["card"], c.get("engine") or engine(c["commit"]), c["host_threads"]))
        if c.get("aiclk"):
            clk[n].append(c["aiclk"]["median"])
        load[n].append(c["load"]["max"])
    rt = {n: statistics.median(ts) for n, ts in timed.items()}
    sigma_rung = MODEL_SIGMA_RUNG.get(model, SIGMA_RUNG)
    s = timed.get(sigma_rung, [])
    sigma = statistics.stdev(s) / statistics.mean(s) if len(s) > 1 else 0.0
    out = {"sigma": round(sigma, 4), "sigma_reps": s, "runtimes": rt,
           "aiclk": {n: statistics.median(v) for n, v in clk.items()},
           "load_max": {n: max(v) for n, v in load.items()}, "coverage": failed, "rungs": {}}
    all_fit = [n for n in rt if sb.FIT_LO <= n <= sb.FIT_HI]
    ref = QUIET_SIGMA.get(model)
    quotable = ref is None or sigma <= 3 * ref
    out["quiet_sigma"] = ref
    for n in sorted(r for r in {*rt, *failed} if r > sb.FIT_HI):
        if n in failed:
            out["rungs"][n] = {"verdict": "COVERAGE", "why": failed[n][:300]}
            continue
        if len(s) < 2:
            out["rungs"][n] = {"verdict": "VOID", "why": f"sigma rung {sigma_rung} has {len(s)} quiet timed folds, needs 2"}
            continue
        fit_rungs = all_fit
        if model in CLOCK_MATCHED_FIT and n in out["aiclk"]:
            fit_rungs = [r for r in all_fit if r in out["aiclk"]
                         and abs(out["aiclk"][r] - out["aiclk"][n]) <= sb.CLOCK_TOL * out["aiclk"][n]]
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
        if not quotable and v["verdict"] in ("PASS", "FAIL"):
            v = {**v, "verdict": "NOT QUOTABLE", "bar_verdict": v["verdict"],
                 "why": f"fit sigma {sigma:.1%} is more than 3x the quiet {ref:.1%}"}
        out["rungs"][n] = {**v, "fit_rungs": fit_rungs, "runtime_s": rt[n], "aiclk": out["aiclk"][n],
                           "load": out["load_max"][n]}
    return out


def main(argv):
    runs = Path(argv[1]) if len(argv) > 1 else ROOT / "perf" / "mgx-speed" / "runs"
    for f in sorted(runs.glob("*.jsonl")):
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        print(json.dumps({"model": f.stem, **model_verdicts(f.stem.split(".")[0], lines)}, default=str))


if __name__ == "__main__":
    main(sys.argv)
