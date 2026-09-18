#!/usr/bin/env python3
"""Probe one model's 256 aa size-ladder rung N times on this card, clock sampled DURING each fold.

The recorded p300c 256 cell is a single draw (the baseline entry says ``reps: 1``) and the rung is
bimodal: 6.5 s against 9.0 s for openbind, on both arms of a lever A/B, so it is host state. A
record pass that happens to re-draw the slow mode reproduces the cell it is replacing, so the mode
gets measured before anything is recorded.

The fold is ``release_gate._run_census_fold`` itself, not a copy of its command line: probe and
record then differ in nothing but how many reps they keep, and the numbers are comparable to the
baseline cell by construction.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:<slug> \
        python3 perf/c14_p300c_256/probe256.py --model openfold3 --card 2 --reps 3
"""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Both, and REPO_ROOT first: running a script BY PATH puts the script dir on sys.path[0],
# not the cwd, so without this the probe imports tt_bio from the editable install
# (/home/ttuser/tt-bio-dev, 1641 commits behind main) instead of the tree under test.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))
import release_gate as rg  # noqa: E402


class ClockSampler:
    """tt_aiclk at 4 Hz in a thread. A number without a clock is not a measurement, and the
    clock has to be read while the fold runs: on this box the governor, not the fold, sets the
    time, and an idle read carries the residue of whatever ran last."""

    def __init__(self, card: int, hz: float = 4.0):
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        self.dt = 1.0 / hz
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.monotonic(), int(self.path.read_text().strip())))
            except Exception:
                pass
            self._stop.wait(self.dt)

    def __enter__(self):
        if not self.path.exists():
            sys.exit(f"no {self.path} -- wrong card number?")
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)

    def window(self, t0: float, t1: float) -> dict:
        v = [c for t, c in self.samples if t0 <= t <= t1]
        if not v:
            return {"n": 0}
        return {"n": len(v), "min": min(v), "median": int(statistics.median(v)), "max": max(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rung", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    workdir = REPO_ROOT / "perf" / "c14_p300c_256" / f"work_{a.model}_{a.rung}"
    workdir.mkdir(parents=True, exist_ok=True)
    reps = []
    with ClockSampler(a.card) as clk:
        # rep 0 is discarded, same as the recorder: the JIT kernel cache is keyed by shape, so
        # the first fold at a rung compiles it.
        for rep in range(a.reps + 1):
            tag = "warmup" if rep == 0 else f"probe{rep - 1}"
            t0 = time.monotonic()
            r = rg._run_census_fold(a.model, a.rung, workdir, tag)
            t1 = time.monotonic()
            if r.get("error") or r.get("refused"):
                print(f"{tag}: {r.get('error') or r.get('refused')}", flush=True)
                return 1
            row = {"tag": tag, "runtime_s": r["runtime_s"], "wall_s": round(r["wall"], 1),
                   "grid": r.get("grid"), "clock_mhz": clk.window(t0, t1),
                   "runtime_src": r.get("runtime_src")}
            reps.append(row)
            print(f"{tag:8s} runtime_s={row['runtime_s']:7.2f} wall={row['wall_s']:7.1f} "
                  f"grid={row['grid']} clock={row['clock_mhz']}", flush=True)

    kept = [r["runtime_s"] for r in reps[1:]]
    out = {"model": a.model, "rung": a.rung, "card": a.card, "reps": reps,
           "kept_runtime_s": kept,
           "min_s": min(kept), "median_s": round(statistics.median(kept), 2),
           "max_s": max(kept),
           "grid": reps[0]["grid"],
           "clock_mhz_all": clk.window(0, time.monotonic())}
    print(f"\n{a.model}/{a.rung}: min {out['min_s']:.2f} median {out['median_s']:.2f} "
          f"max {out['max_s']:.2f} s over {len(kept)} kept reps, grid {out['grid']}, "
          f"clock {out['clock_mhz_all']}")
    path = Path(a.out) if a.out else workdir.parent / f"probe_{a.model}_{a.rung}_card{a.card}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
