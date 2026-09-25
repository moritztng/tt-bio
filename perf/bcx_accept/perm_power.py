#!/usr/bin/env python3
"""Which comparison can actually separate the two arms at the trajectory counts BCX can buy.

DECISION-RULE.md's two clauses are both degenerate at this n, in opposite directions:

  rate        1/10 device against 1/2 reference -- the Clopper-Pearson intervals overlap for
              any true rates whatsoever, so it returns "consistent" whatever the loop does.
              Separating 0.167 from 0.33 at 80 % power needs ~109 trajectories an arm.
  containment a device median inside the reference's observed per-stage RANGE. Under the null
              a fresh draw lands inside the range of n earlier draws with probability exactly
              (n-1)/(n+1) -- 1/3 at the reference's n=2. It returns "outside" whatever the
              loop does, and needs n=39 for 95 % specificity.

Neither is evidence. This is the third option and it is the only one whose resolution improves
with the trajectories already running.

Take one number per trajectory -- a within-trajectory statistic, so the trajectory is its own
control and a different binder draw does not by itself move it -- and compare the two arms with
an EXACT two-sample permutation test over all C(n_dev+n_ref, n_ref) relabelings.

The power statement that matters, and it is the reason this instrument is worth building:
with 8 device and 2 reference trajectories there are C(10,2) = 45 relabelings, so the smallest
attainable one-sided p is 1/45 = 0.022. A real difference CAN clear 0.05 at today's n. That is
not true of either clause above. Each further reference trajectory roughly triples the resolution
(C(11,3) = 165, C(12,4) = 495).

What it does NOT do: a significant p here says the two arms' distributions differ on that
statistic, not that the device is worse, and with n_ref = 2 a single atypical reference
trajectory drives everything. Report p beside the two sets of raw numbers, never alone.
"""
import csv
import re
import statistics as st
import sys
from itertools import combinations

# Stage boundaries are round counts, not a column: the losses CSV carries no stage label.
# screen 50 / refine 25 / anneal 45 / harden 5 (arm_stamp.json stage_plan), 125 gradient rounds.
PLAN = (("screen", 50), ("refine", 25), ("anneal", 45), ("harden", 5))
RE_DRAW = re.compile(r"_l(\d+)_([0-9a-f]+)_losses\.csv$")


def rounds(path, col="hPDL1.iptm"):
    out = []
    for row in csv.DictReader(open(path, newline="")):
        v = (row.get(col) or "").strip()
        if v:
            out.append(float(v))
    return out


def stages(v):
    out, i = {}, 0
    for name, n in PLAN:
        out[name] = v[i:i + n]
        i += n
    return out


def jitter(v):
    d = [abs(b - a) for a, b in zip(v, v[1:])]
    return st.median(d) if d else None


STATS = {}


def stat(fn):
    STATS[fn.__name__.replace("_", " ")] = fn
    return fn


@stat
def gradient_jitter(s, p, t):
    return jitter([x for n, _ in PLAN for x in s[n]])


@stat
def anneal_peak_to_end(s, p, t):
    a = s["anneal"]
    return max(a) - max(a[-5:]) if len(a) >= 5 else None


@stat
def anneal_harden_step(s, p, t):
    a, h = s["anneal"], s["harden"]
    return max(h) - max(a[-5:]) if h and len(a) >= 5 else None


@stat
def iptm_over_ptm_jitter(s, p, t):
    ji = jitter([x for n, _ in PLAN for x in s[n]])
    jp = jitter([x for n, _ in PLAN for x in p[n]])
    return None if not jp else ji / jp


@stat
def mean_iptm_gradient(s, p, t):
    v = [x for n, _ in PLAN for x in s[n]]
    return st.fmean(v) if v else None


@stat
def fraction_rounds_collapsed(s, p, t):
    """Share of gradient rounds with i_pTM below 0.25. The bistability ref_s1's l111 showed in
    its anneal tail is a property of the whole run, not of the stage it finally died in."""
    v = [x for n, _ in PLAN for x in s[n]]
    return sum(x < 0.25 for x in v) / len(v) if v else None


def perm_p(dev, ref):
    """Exact two-sided p on |median difference| over every relabeling."""
    pool = dev + ref
    k = len(ref)
    obs = abs(st.median(dev) - st.median(ref))
    hit = tot = 0
    for idx in combinations(range(len(pool)), k):
        r = [pool[i] for i in idx]
        d = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        if abs(st.median(d) - st.median(r)) >= obs - 1e-12:
            hit += 1
    return hit / tot, tot


def main(argv):
    side, specs = None, []
    for a in argv[1:]:
        if a in ("--device", "--reference"):
            side = a[2:]
        else:
            label, _, path = a.partition("=")
            specs.append((label, side, path))

    data = {}
    for label, s, path in specs:
        iptm, ptm = stages(rounds(path)), stages(rounds(path, "hPDL1.ptm"))
        data.setdefault(s, []).append((label, iptm, ptm))

    n_dev, n_ref = len(data.get("device", [])), len(data.get("reference", []))
    from math import comb
    print(f"\n# Exact permutation test, {n_dev} device against {n_ref} reference trajectories")
    print(f"# {comb(n_dev + n_ref, n_ref)} relabelings, so the smallest attainable p is "
          f"{1 / comb(n_dev + n_ref, n_ref):.4f}\n")
    print(f"{'statistic':<26}{'device (median)':>34}{'reference':>22}{'p':>8}")
    print("-" * 92)
    for name, fn in STATS.items():
        vals = {}
        for s in ("device", "reference"):
            vals[s] = [(lbl, fn(i, p, lbl)) for lbl, i, p in data.get(s, [])]
            vals[s] = [(lbl, v) for lbl, v in vals[s] if v is not None]
        dev = [v for _, v in vals["device"]]
        ref = [v for _, v in vals["reference"]]
        if len(dev) < 2 or len(ref) < 1:
            continue
        p, _ = perm_p(dev, ref)
        ds = f"{st.median(dev):+.3f}  [{min(dev):+.3f}, {max(dev):+.3f}] n={len(dev)}"
        rs = f"{st.median(ref):+.3f}  n={len(ref)}"
        print(f"{name:<26}{ds:>34}{rs:>22}{p:>8.3f}")
        print(f"{'':<26}reference raw: "
              + ", ".join(f"{lbl} {v:+.3f}" for lbl, v in vals['reference']))
    print("\nA p at or below 0.05 needs the reference trajectories to be extreme within the pool.")
    print("With n_ref = 2 one atypical reference run drives every column -- read the raw values.")


if __name__ == "__main__":
    main(sys.argv)
