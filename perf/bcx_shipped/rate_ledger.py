#!/usr/bin/env python3
"""One row per COMPLETED trajectory, with the seconds it cost, across every acceptance arm.

DECISION-RULE.md: a rate or a cost per accepted design divides by every completed
trajectory, never only by the accepting ones. A BindCraft 2 rejection is not cheap --
it arrives after most of the gradient work -- so leaving the rejections out of the
denominator understates the cost per design several-fold.

Two things this is careful about, and the second is the one that gets misquoted:

1. A TRAJECTORY'S BOUNDARY IS THE LAST WRITE INTO ITS OWN DIRECTORY, not an arm stamp.
   `arm_stamp.json` records `arm exit 0` even for a run that died in the validation
   ensemble, so the stamp cannot tell a completion from a crash.

2. THESE ARE CONTENDED WALL SECONDS ON ONE CHIP, NOT QUIET CHIP-SECONDS. The BC2
   gradient loop is 79.2 % host work and qb2 carried 3-4 arms on 16 cores throughout,
   at a load median of 14.7-17.3. So every figure here is an UPPER BOUND on what the
   same trajectory costs on a quiet box, and the ratio against another workload's
   number is only fair if that number was measured under the same contention. Say
   "upper bound" wherever this is quoted.

Usage:
  rate_ledger.py ARM=ARTDIR@START[:END] [...] [--reference K/N]   ISO8601 Z

--reference gives the other arm's accepted/completed counts and adds a Fisher exact test of
whether the two acceptance counts differ at all. That is the comparison the GO condition asks
for, and at the counts this campaign can buy it is the honest form of it: a point-estimate
ratio between 1/10 and 1/2 invites a conclusion the interval does not support.

Give END for an arm that has exited. The MPNN redesign and validation ensemble run AFTER
the last write into the trajectory's own directory, so without END an accepted trajectory
is charged only for its gradient half -- route_s3's l73 reads 4,350 s against the arm's
own 5,137 s wall. The residual END - (last trajectory boundary) is that trajectory's
validation work and is added to it.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

RE_DRAW = re.compile(r"_l(\d+)_([0-9a-f]+)$")
BOLTZGEN_CHIP_S = 264.3   # state/cmp/LOAD.md, the workload CMP ranks against


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval via the beta quantile, no scipy on this box."""
    def beta_q(p, a, b):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if betainc(a, b, mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def betainc(a, b, x):
        # regularized incomplete beta by continued fraction (Lentz)
        import math
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 300):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1.0 - c * d) < 1e-12:
                break
        return front * (f - 1.0)

    lo = 0.0 if k == 0 else beta_q(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_q(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def fisher_exact(a, b, c, d):
    """Two-sided p on a 2x2 by summing every table no more probable than the observed one.
    No scipy on this box, and the counts are small enough to enumerate exactly."""
    from math import comb
    n = a + b + c + d

    def p_tab(a, b, c, d):
        return comb(a + b, a) * comb(c + d, c) / comb(n, a + c)

    obs = p_tab(a, b, c, d)
    tot = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        j, k = a + b - i, a + c - i
        l = c + d - k
        if min(j, k, l) < 0:
            continue
        pr = p_tab(i, j, k, l)
        if pr <= obs + 1e-12:
            tot += pr
    return tot


def traj_end(d):
    """Last write into the trajectory's own directory."""
    best = 0.0
    for root, _, files in os.walk(d):
        for f in files:
            best = max(best, os.path.getmtime(os.path.join(root, f)))
    return best


def terminal_map(log):
    """draw hash -> terminal label, read off the run log the same way stage_profile does."""
    out, cur = {}, None
    for raw in open(log, errors="replace"):
        line = re.sub(r"^\d{4}-\d\d-\d\dT[\d:]+Z ", "", raw).rstrip()
        m = re.search(r"=== trajectory \d+ \| (\S+) \| accepted", line)
        if m:
            cur = RE_DRAW.search(m.group(1))
            cur = cur.group(2) if cur else m.group(1)
            out[cur] = "in flight"
            continue
        if cur is None:
            continue
        m = re.search(r"rejected at (\w+) design stage.*?due to \[([^\]]*)\]", line)
        if m:
            out[cur] = f"{m.group(1)} [{m.group(2)}]"
        elif re.search(r"trajectory rejected.*?due to \[([^\]]*)\]", line):
            out[cur] = f"final [{re.search(r'due to .([^]]*)', line).group(1)}]"
        else:
            m = re.search(r"(\d+) of (\d+) redesigns passed", line)
            if m:
                # "0 of 10 redesigns passed" is what BC2 prints for a trajectory the refold
                # ensemble REJECTED. Read the count, never just the line -- see stage_profile.py.
                out[cur] = "ACCEPTED" if int(m.group(1)) else f"validation [0 of {m.group(2)}]"
    return out


def csv_terminal_map(artdir):
    """draw hash -> terminal label, read off BindCraft 2's OWN csvs rather than the run log.

    The log is not a durable record. `qb1_launch.sh:84` opens `$TAG.log` with a truncating
    redirect, so re-running the launcher for a tag whose arm has already finished destroys
    that arm's whole stage trace -- which is what happened to `qb1_s10` at 2026-09-26T09:14:14Z
    and to `qb1_s11` at 01:39. Both logs are 73 bytes of `line 88: d: command not found`.

    `!_Trajectories.csv` carries a `terminated` column that BC2 writes per trajectory, and
    `!_Refolded.csv` carries one row per redesign with an `outcome`. Between them the terminal
    is recoverable with no log at all. The csv loses the FILTER LIST, so the log is still
    worth reading when it survives -- as an enrichment, not as the source of truth.

    An empty `terminated` means the trajectory ran the full design path into the refold
    ensemble. That is NOT acceptance: `terminated: completed` counts l69 and l180 alike and
    only `!_Refolded.csv` separates them. Count the passing rows.
    """
    import csv as _csv
    from collections import Counter
    out = {}
    tpath = os.path.join(artdir, "1_Trajectories", "!_Trajectories.csv")
    if not os.path.exists(tpath):
        return out
    passed, scored = Counter(), Counter()
    rpath = os.path.join(artdir, "2_Refolded", "!_Refolded.csv")
    if os.path.exists(rpath):
        for r in _csv.DictReader(open(rpath)):
            m = RE_DRAW.search(re.sub(r"_candidate\d+$", "", r.get("design", "")))
            if not m:
                continue
            scored[m.group(2)] += 1
            if r.get("outcome", "").strip().lower() == "passed":
                passed[m.group(2)] += 1
    for r in _csv.DictReader(open(tpath)):
        draw = (r.get("hash") or "").strip()
        if not draw:
            continue
        stage = (r.get("terminated") or "").strip()
        if stage:
            out[draw] = stage
        elif passed[draw]:
            out[draw] = "ACCEPTED"
        elif scored[draw]:
            out[draw] = "validation [0 of %d]" % scored[draw]
        else:
            out[draw] = "in flight"
    return out


def merge_terminals(csv_term, log_term, label):
    """csv is the source of truth; the log only supplies the filter list it carries.

    Disagreement is reported rather than silently resolved -- the two are independent
    records and a split between them means one of them is describing a different run.
    """
    out = dict(csv_term)
    for draw, lt in log_term.items():
        ct = csv_term.get(draw)
        if ct is None:
            out[draw] = lt
            continue
        if lt == "in flight" or ct == "in flight":
            continue
        base = lt.split(" [")[0]
        if base == ct or (base == "final" and ct == "final") or (
                lt == "ACCEPTED" and ct == "ACCEPTED") or (
                lt.startswith("validation") and ct.startswith("validation")):
            out[draw] = lt   # same verdict, log adds the filter list
        else:
            print("  ! %s %s: csv says %r, log says %r -- using the csv"
                  % (label, draw[:8], ct, lt), file=sys.stderr)
    return out


def main(argv):
    rows, ref = [], None
    argv = list(argv)
    if "--reference" in argv:
        i = argv.index("--reference")
        k, _, n = argv[i + 1].partition("/")
        ref = (int(k), int(n))
        del argv[i:i + 2]
    for spec in argv[1:]:
        label, _, rest = spec.partition("=")
        artdir, _, start = rest.partition("@")
        # START[:END] -- both are ISO stamps and carry colons, so split on the separator
        start_s, _, arm_end_s = start.partition(":2026-")
        arm_end_s = ("2026-" + arm_end_s) if arm_end_s else ""
        t = datetime.strptime(start_s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
        arm_end = (datetime.strptime(arm_end_s, "%Y-%m-%dT%H:%M:%SZ")
                   .replace(tzinfo=timezone.utc).timestamp()) if arm_end_s else None
        log = artdir.rstrip("/") + ".log"
        if not os.path.exists(log):
            log = os.path.join(os.path.dirname(artdir.rstrip("/")),
                               os.path.basename(artdir.rstrip("/")).replace("_arm", "")
                               + ".log")
        csv_term = csv_terminal_map(artdir)
        if os.path.exists(log) and os.path.getsize(log) > 200:
            term = merge_terminals(csv_term, terminal_map(log), label)
        else:
            # log lost or never written; BC2's own csvs still carry every terminal
            print("  ! %s: no usable log at %s, reading terminals from the csvs only"
                  % (label, log), file=sys.stderr)
            term = csv_term
        order = json.load(open(os.path.join(artdir, ".campaign_state.json")))["attempted"]
        dirs = {}
        for d in os.listdir(os.path.join(artdir, "1_Trajectories")):
            m = RE_DRAW.search(d)
            if m:
                dirs[m.group(2)] = (int(m.group(1)),
                                    traj_end(os.path.join(artdir, "1_Trajectories", d)))
        prev, mine = t, []
        for draw in order:
            if draw not in dirs:
                continue
            length, end = dirs[draw]
            mine.append([label, f"l{length}", draw, term.get(draw, "?"), end - prev, end])
            prev = end
        # charge the arm's last completed trajectory for the validation work that ran after
        # the last write into its directory
        done_here = [r for r in mine if r[3] not in ("?", "in flight")]
        if arm_end and done_here and arm_end > done_here[-1][5]:
            done_here[-1][4] += arm_end - done_here[-1][5]
            done_here[-1][5] = arm_end
        rows += [tuple(r) for r in mine]

    done = [r for r in rows if r[3] not in ("?", "in flight")]
    acc = [r for r in done if r[3] == "ACCEPTED"]
    print("\n# Completed-trajectory ledger -- CONTENDED wall seconds on one chip, upper bound\n")
    print(f"{'arm':<12}{'draw':<7}{'terminal':<24}{'seconds':>9}   ended")
    print("-" * 74)
    for label, ln, draw, t_, secs, end in rows:
        if t_ in ("?", "in flight"):
            print(f"{label:<12}{ln:<7}{t_:<24}{'.':>9}   (excluded, in flight)")
            continue
        print(f"{label:<12}{ln:<7}{t_:<24}{secs:>9.0f}   "
              f"{datetime.fromtimestamp(end, timezone.utc):%H:%M:%SZ}")
    total = sum(r[4] for r in done)
    n, k = len(done), len(acc)
    print("-" * 74)
    print(f"{'':<19}{n} completed, {k} accepted{total:>26.0f}")
    if k:
        per = total / k
        lo, hi = clopper_pearson(k, n)
        print(f"\nrate {k}/{n} = {k / n:.3f}   Clopper-Pearson 95 % CI {lo:.4f} - {hi:.4f}")
        print(f"{per:,.0f} s per accepted design, upper bound "
              f"= {per / BOLTZGEN_CHIP_S:.1f}x BoltzGen's {BOLTZGEN_CHIP_S} chip-s/design")
        print(f"CI on the ratio: {total / hi / n / BOLTZGEN_CHIP_S:.1f}x - "
              f"{total / lo / n / BOLTZGEN_CHIP_S:,.0f}x")
        print(f"{3600 * k / total:.3f} designs/h/chip, upper bound on the chip's cost")
        if ref:
            rk, rn = ref
            pv = fisher_exact(k, n - k, rk, rn - rk)
            rlo, rhi = clopper_pearson(rk, rn)
            print(f"\nreference {rk}/{rn} = {rk / rn:.3f}   95 % CI {rlo:.4f} - {rhi:.4f}")
            print(f"Fisher exact two-sided p = {pv:.4f} -- the two acceptance counts "
                  f"{'do not differ' if pv > 0.05 else 'DIFFER'}")
            print(f"the intervals overlap across {max(lo, rlo):.4f} - {min(hi, rhi):.4f}")
    else:
        print(f"\n0 accepted over {n} completed -- lower bound {total:,.0f} s per accepted "
              f"design, resting on {n} trajectories. Not a rate.")


if __name__ == "__main__":
    main(sys.argv)
