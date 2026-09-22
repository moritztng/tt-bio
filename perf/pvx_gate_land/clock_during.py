#!/usr/bin/env python3
"""Report the AICLK each SCORED fold actually ran at, or refuse the cell.

The whole-run min/max is not an answer. A run alternates folding and idling, and on this host a
card sits at 800 MHz between folds and 1350 during them, so min/max over every sample in the run
window reports "min 800 max 1350" for folds that never left 1350. That reading was taken on
2026-09-20 and could not be turned into a reportable number, which is the entire reason this
exists: intersect each fold's own [t_start, t_end] with the sampler's records and score only what
falls INSIDE a fold.

WHY THE WINDOW IS NOT THE SUBPROCESS. fold_ab_flip times a fold as `runtime_s` out of the fold's
own results.json, which EXCLUDES model load and process startup, while [t_start, t_end] brackets
the whole subprocess. On a 352 aa protenix-v2 cell that is a 24-26 s timed fold inside a 33-34 s
subprocess, and the card sits at 800 MHz for the ~8 s of checkpoint load at the front. Scoring the
subprocess therefore reported an in-fold dip to 800 for four of six folds that never dipped while
they were being timed.

Nothing in results.json says where the timed region starts, so this does not guess at it. The
timed fold is a contiguous `runtime_s`-long window somewhere inside the subprocess, so it is
sufficient to find the longest unbroken run of samples at or above --min-mhz and ask whether the
timed fold could fit inside it. Sample spans understate the true run by up to one interval at each
end, so the test errs toward refusing, which is the safe direction for a gate: it can fail a good
cell and tell you to sample faster, but it cannot pass a throttled one.

Refuses, rather than reporting a weaker number, when:
  * the samples do not EVIDENCE the fold. One sample is an instant, not a fold: a 30 s sampler
    against a 26 s fold lands one record inside it and says nothing about the other 25 s. This bar
    is the reason the file exists; an earlier draft required only one sample and a synthetic replay
    of the 2026-09-20 run passed it.
  * the longest unbroken run at or above --min-mhz is shorter than `runtime_s`, so the timed fold
    cannot be placed anywhere that did not throttle. Per the standing clock rule the AICLK sets the
    fold time, and a timing taken through a dip is an artifact of the card, not of the lever.
"""
import argparse, json, sys


def mhz(v):
    """AICLK as a float, or None. tt-smi reports it as a right-aligned STRING (" 800", "1350"),
    so an isinstance(v, (int, float)) filter silently drops every sample and the cell is then
    refused for "no samples" when the samples are all there. Found by running the sampler, not
    by reading it."""
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def longest_ok_run(inside, min_mhz):
    """(span_seconds, n_samples) of the longest unbroken run of samples at or above min_mhz."""
    best = (0.0, 0)
    run = []
    for t, v in inside:
        if v >= min_mhz:
            run.append(t)
        else:
            run = []
        if len(run) >= 2:
            best = max(best, (run[-1] - run[0], len(run)))
        elif len(run) == 1:
            best = max(best, (0.0, 1))
    return best


def fold_clocks(folds, samples, card):
    """[(fold, [(t, aiclk) inside the subprocess])] for each scored fold, in order."""
    out = []
    for f in folds:
        inside = [(t, v) for t, v in samples if f["t_start"] <= t <= f["t_end"]]
        out.append((f, inside))
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", help="fold_ab_flip.py --out json")
    ap.add_argument("contention", help="sample_contention.py jsonl")
    ap.add_argument("card", help="card index the folds were pinned to")
    ap.add_argument("--min-mhz", type=float, default=1200.0)
    ap.add_argument("--min-span-frac", type=float, default=0.5,
                    help="fraction of a fold the in-fold samples must span to evidence it")
    a = ap.parse_args()

    cells = json.load(open(a.artifact))["cells"]
    samples, load = [], []
    for line in open(a.contention):
        r = json.loads(line)
        if "t" not in r:
            continue  # pre-2026-09-21 record, no epoch field to intersect on
        v = mhz((r.get("aiclk") or {}).get(a.card))
        if v is not None:
            samples.append((r["t"], v))
        load.append((r["t"], r["loadavg"][0]))

    bad = []
    for c in cells:
        folds = c.get("folds") or []
        if not folds:
            bad.append(f"{c['model']} {c['rung']}aa: artifact carries no per-fold intervals")
            continue
        print(f"{c['model']} {c['rung']}aa, card {a.card}, AICLK inside each scored fold:")
        for f, inside in fold_clocks(folds, samples, a.card):
            label = f"  {f['arm']:<3} rep{f['rep']}  {f['runtime_s']:7.4f}s"
            sub = f["t_end"] - f["t_start"]
            span = (inside[-1][0] - inside[0][0]) if len(inside) > 1 else 0.0
            if len(inside) < 2 or span < a.min_span_frac * sub:
                print(f"{label}  {len(inside)} samples spanning {span:.0f}s of a {sub:.0f}s "
                      f"subprocess -- DOES NOT EVIDENCE THIS FOLD")
                bad.append(f"{c['model']} {c['rung']}aa {f['arm']} rep{f['rep']}: "
                           f"{len(inside)} sample(s) spanning {span:.0f}s of a {sub:.0f}s "
                           f"subprocess, under {a.min_span_frac:.0%}; lower --interval")
                continue
            ok_span, ok_n = longest_ok_run(inside, a.min_mhz)
            hi = max(v for _, v in inside)
            print(f"{label}  {len(inside)} samples over a {sub:.0f}s subprocess; longest run at "
                  f"or above {a.min_mhz:.0f} MHz is {ok_span:.0f}s ({ok_n} samples), peak "
                  f"{hi:.0f} MHz")
            if ok_span < f["runtime_s"]:
                bad.append(f"{c['model']} {c['rung']}aa {f['arm']} rep{f['rep']}: longest "
                           f"unthrottled run {ok_span:.0f}s cannot contain a {f['runtime_s']:.1f}s "
                           f"timed fold; sample faster or the card throttled")
        inside_load = [l for t, l in load
                       if any(f["t_start"] <= t <= f["t_end"] for f in folds)]
        if inside_load:
            print(f"  loadavg1 inside the folds: min {min(inside_load):.2f} "
                  f"max {max(inside_load):.2f}")

    if bad:
        print("\nREFUSING the cell, the timings above are not reportable:")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\nEvery scored fold has a during-sampled clock at or above "
          f"{a.min_mhz:.0f} MHz. The timings are reportable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
