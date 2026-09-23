#!/usr/bin/env python3
"""How many designs per size separate 1536 from 512, and why more designs is not enough.

The row opened on one design at each size, 4.054 A against 11.480 A, and the first thing this
campaign measured is that a single 512 target spans 3.30-13.78 A across eight draws. So "is
1536 worse" is a question about two distributions, and the honest form of the answer is a
sample size with a detectable effect attached to it.

Two different floors have to be cleared and they need different things:

  * **within a target**, the design-to-design spread. More designs fixes this, and the table
    below says how many for how big an effect. Resampled from the designs actually measured,
    not from an assumed shape -- the distribution is bimodal-ish (a few good designs, a tail
    of failures) and no normal-theory formula describes it.
  * **across targets**, the fact that a ladder changes the binding problem along with the
    size. More designs does NOT fix this. Two 512 targets measured here have medians 2.10 A
    apart, which is most of the gap the row set out to explain, and with one target per size
    the size effect and the target effect are the same number.

    python3 perf/mgxaccuracy/power.py perf/mgxaccuracy/results/*.jsonl
"""
import argparse
import json
import pathlib
import random
import statistics as st
import sys

ALPHA_Z = 1.96  # two-sided 0.05, normal approximation to Mann-Whitney U
TRIALS = 4000


def mw_reject(a, b) -> bool:
    """Two-sided Mann-Whitney at 0.05, tie-aware ranks, normal approximation.

    The same test `report.py` prints, so the power stated here is the power of the test the
    verdict will actually be read off."""
    pool = sorted(a + b)
    ranks = {}
    i = 0
    while i < len(pool):
        j = i
        while j + 1 < len(pool) and pool[j + 1] == pool[i]:
            j += 1
        r = (i + j) / 2 + 1
        ranks[pool[i]] = r
        i = j + 1
    na, nb = len(a), len(b)
    ra = sum(ranks[v] for v in a)
    u = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = (na * nb * (na + nb + 1) / 12) ** 0.5
    return sd > 0 and abs(u - mu) / sd > ALPHA_Z


def power(sample, shift, n, trials=TRIALS, rng=None) -> float:
    """P(reject) when the 1536 arm is the 512 distribution scaled by `shift`.

    Multiplicative rather than additive because scRMSD is a positive quantity whose spread
    grows with its level: a +2 A shift on a design that refolds at 0.86 A is a different
    claim from the same shift at 12.9 A."""
    rng = rng or random.Random(0)
    hits = 0
    for _ in range(trials):
        a = [rng.choice(sample) for _ in range(n)]
        b = [rng.choice(sample) * shift for _ in range(n)]
        hits += mw_reject(a, b)
    return hits / trials


def load_512(paths) -> dict:
    """Every n>=2 boltzgen scRMSD sample at 512, keyed by target."""
    out = {}
    for p in paths:
        p = pathlib.Path(p).expanduser()
        if not p.is_file() or p.suffix != ".jsonl":
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            d = r.get("dsg") or r
            sc = d.get("scrmsd")
            if not sc or len(sc) < 2 or r.get("model") != "boltzgen":
                continue
            if r.get("target_res") != 512:
                continue
            key = (r.get("target") or f"?{p.name}").rsplit("/", 1)[-1]
            out.setdefault(key, []).extend(float(v) for v in sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--shifts", default="1.15,1.25,1.5,2.0,2.8")
    ap.add_argument("--ns", default="4,8,12,16,24,32,48,64")
    args = ap.parse_args()

    per_target = load_512(args.jsonl)
    if not per_target:
        print("no n>=2 boltzgen 512 sample found", file=sys.stderr)
        return 1
    pooled = [v for vs in per_target.values() for v in vs]

    print(f"\nthe 512 distribution these numbers come from ({len(pooled)} designs, "
          f"{len(per_target)} target(s))")
    for k, vs in sorted(per_target.items()):
        print(f"  {k[:40]:<42} n={len(vs):<3} median {st.median(vs):6.2f}  "
              f"range {min(vs):5.2f}-{max(vs):5.2f}")
    print(f"  {'POOLED':<42} n={len(pooled):<3} median {st.median(pooled):6.2f}  "
          f"range {min(pooled):5.2f}-{max(pooled):5.2f}")

    shifts = [float(x) for x in args.shifts.split(",")]
    ns = [int(x) for x in args.ns.split(",")]
    print(f"\npower to reject at 0.05 two-sided, {TRIALS} resamples, per size"
          f"\n(columns: the 1536 arm is the 512 distribution times this factor)")
    print(f"\n{'n per size':>11}" + "".join(f"{s:>9.2f}x" for s in shifts))
    rng = random.Random(20260923)
    table = {}
    for n in ns:
        row = [power(pooled, s, n, rng=rng) for s in shifts]
        table[n] = row
        print(f"{n:>11}" + "".join(f"{v*100:>9.0f}%" for v in row))

    print("\nread it as: the smallest n whose row clears 80 %")
    for i, s in enumerate(shifts):
        ok = [n for n in ns if table[n][i] >= 0.80]
        print(f"  {s:>4.2f}x median shift -> "
              + (f"n = {ok[0]} designs per size" if ok else
                 f"more than {ns[-1]} designs per size"))

    # The other floor. This is arithmetic on measured medians, not a simulation: with one
    # target per size there is no way to tell the two effects apart, whatever n is.
    meds = sorted(st.median(v) for v in per_target.values())
    print("\nand the floor that more designs does not move")
    if len(meds) < 2:
        print("  only one 512 target measured — the across-target floor is unmeasured, and "
              "until it is,\n  any size effect is indistinguishable from a target effect")
    else:
        print(f"  {len(meds)} targets at 512, medians "
              f"{' / '.join(f'{m:.2f}' for m in meds)} -> spread {meds[-1]-meds[0]:.2f} A")
        print(f"  A 1536 median inside {meds[0]:.2f}-{meds[-1]:.2f} A is not evidence of a "
              f"size effect at ANY n,\n  because two targets of the SAME size already differ "
              f"by that much. Clearing it needs\n  targets, not designs: at least three per "
              f"size, paired so each target is measured at both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
