#!/usr/bin/env python3
"""Per-rung host load during a size-ladder pass, joined from the fold logs and the sampler.

WHY THIS EXISTS. `drive.sh` gates on the loadavg at the moment it starts, and the gate's own
`--load-ceiling` does the same. Both are blind to load CHANGING during the ladder, and a ladder
is a sequence of rungs measured minutes apart, so a box that empties out halfway through times
its small rungs under contention and its large rungs quiet. That does not average out -- it
bends the curve the arm scores.

Measured, not argued: protenix-v1 recorded 2026-09-20 12:09-12:21Z as load fell 19.7 -> 10.8 and
came out 256 6.4->9.9 (1.55x), 512 13.4->18.9 (1.41x), 640 1.05x, 768 1.04x, 896 1.01x,
1024 1.07x. The lever census was clean and every rung above 512 reproduced, so nothing about the
model changed; the first two rungs simply ran on a different machine than the last four. The
recorded exponents came out 0.933 and 1.013 against the previous 1.066 and 1.773, and a later
quiet check reading 1.77 would miss the recorded 1.01 by 0.76 against a +-0.50 band. Committing
that is punching a hole in the gate, not refreshing it.

Sigma cannot see this and never could: `SIZE_LADDER_SIGMA_REPS` takes five folds back to back at
one rung, which measures within-burst noise (2-3 %) while the quantity that moves the verdict is
between-RUNG drift (up to 55 % here). Same wrong-scope defect the 09-15 campaign found between
passes, one level down.

HOW. Every census fold leaves `perf/sizegate/work/<model>-<rung>-rep<N>.log`, whose mtime is when
that fold finished. `perf/pvx_gate_land/sample_contention.py` writes loadavg and per-card AICLK
every 30 s. Joining the two gives the load each rung was actually timed under, after the fact,
with no extra device time.

--window IS NOT OPTIONAL IN PRACTICE. The record and the check of one model write the SAME fold
log paths in the same `--keep` work dir, so a check overwrites the record's mtimes rung by rung
while it runs. Without a window the join silently mixes the two passes: protenix-v1's 256 rung
read a load span of 9.0-28.5 because rep0 and warmup had already been rewritten by the check.
Take the window from the drive log's own start/end lines.

Usage:  rung_load.py [--sampler PATH] [--window FROM TO] [--max-spread 1.6] <model> ...
        FROM/TO are ISO8601 UTC, e.g. 2026-09-20T12:09:00Z
"""
import argparse
import glob
import json
import os
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "perf" / "sizegate" / "work"
DEFAULT_SAMPLER = ROOT / "perf" / "sizeladder_0920" / "logs" / "contention.jsonl"
FOLD = re.compile(r"^(?P<model>.+)-(?P<rung>\d+)-(?:rep\d+|warmup)\.log$")


def samples(path):
    out = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        ts = d.get("ts", "")
        try:
            import datetime
            t = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            continue
        out.append((t, d.get("loadavg", [None])[0], d.get("aiclk") or {}))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", default=str(DEFAULT_SAMPLER))
    ap.add_argument("--window", nargs=2, metavar=("FROM", "TO"),
                    help="ISO8601 UTC bounds; fold logs outside are another pass's")
    ap.add_argument("--max-spread", type=float, default=1.25,
                    help="largest tolerated ratio of per-rung median load across the gated "
                         "rungs 256/512/768 before the pass is called unusable. 1.25 rather "
                         "than something looser because protenix-v1 came out unusable at a "
                         "measured 1.51x: its 256 rung inflated 1.55x and its 512 1.41x while "
                         "640 and up reproduced inside 7 %, which bent k512->768 from 1.773 to "
                         "1.013 -- 0.76 outside the +-0.50 band a later quiet check would read")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    import datetime

    def iso(x):
        return datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()

    w0, w1 = (iso(args.window[0]), iso(args.window[1])) if args.window else (0.0, 1e18)
    s = samples(args.sampler)
    if not s:
        print(f"no samples in {args.sampler} -- cannot say what load any rung saw")
        return 2
    print(f"{len(s)} samples, {s[0][0]:.0f}..{s[-1][0]:.0f}")

    bad = 0
    for model in args.models:
        per = {}
        for f in glob.glob(str(WORK / f"{model}-*.log")):
            m = FOLD.match(os.path.basename(f))
            if not m or m["model"] != model:
                continue
            mt = os.path.getmtime(f)
            if not (w0 <= mt <= w1):
                continue
            per.setdefault(int(m["rung"]), []).append(mt)
        if not per:
            print(f"{model}: no fold logs under {WORK} inside the window")
            continue
        print("=" * 84)
        print(f"{model}")
        med = {}
        for rung in sorted(per):
            lo, hi = min(per[rung]), max(per[rung])
            loads = [l for t, l, _ in s if lo - 60 <= t <= hi + 5 and l is not None]
            clks = sorted({c for t, _, a in s if lo - 60 <= t <= hi + 5
                           for c in a.values() if c and c.strip() != "800"})
            if not loads:
                print(f"  {rung:>5}  no sampler coverage")
                continue
            med[rung] = statistics.median(loads)
            print(f"  {rung:>5}  folds {len(per[rung]):>2}  load med {med[rung]:5.1f} "
                  f"min {min(loads):5.1f} max {max(loads):5.1f}   non-idle aiclk {clks}")
        gated = [med[r] for r in (256, 512, 768) if r in med]
        if len(gated) == 3:
            spread = max(gated) / max(min(gated), 0.01)
            ok = spread <= args.max_spread
            print(f"  gated rungs 256/512/768 load spread {spread:.2f}x "
                  f"({'USABLE' if ok else 'UNUSABLE -- re-record'}; bar {args.max_spread}x)")
            bad += 0 if ok else 1
        else:
            print("  gated rungs not all covered -- cannot judge the exponents")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
