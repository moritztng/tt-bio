#!/usr/bin/env python3
"""Score the narrow-q legs on FOREIGN cpu instead of raw loadavg1.

WHY THE LOADAVG CEILING CANNOT SCORE THIS CELL. narrowq_bank.py cuts a leg whose loadavg1
exceeded 2.00 anywhere inside it. That bar is applied to a quantity that INCLUDES the measured
fold. On 2026-09-22 23:03:50Z this row took a pair on a box whose only CPU consumers, at the
3.68 peak, were `tt_bio.main predict` out of this very worktree and its own multiprocessing
children at 276 % and 229 %. Both legs were cut. The pre-flight had passed at 1.52 against the
same 2.00 number measured WITHOUT the fold, so the guard admitted a window its own scorer was
arithmetically certain to refuse: on the one clean cell (pre-flight 0.13, during max 1.33) the
fold's own contribution is ~1.2, and loadavg1 is a 60 s EWMA that keeps climbing the longer the
process runs, so the second leg of a pair inherits the first leg's tail and clears 2.00 on an
empty box. A contention bar that counts the measurement as contention cuts the legs it exists
to protect.

WHAT THIS SCORES INSTEAD. Contention is other people's load. Per leg it sums the cpu of every
sampled process positively identified as neither this harness nor known always-on infra, and
reports the worst such sample.

THE BAR IS NOT SELF-SERVING, AND HERE IS THE CONTROL. A looser-looking bar proposed by the row
it unblocks has to be shown to still refuse what it should. The retake's reps 2-4 were cut by a
real co-tenant: of3t began stacking ref_grad.py, c64_score.py and model_scope.py at 22:11:40Z.
If this bar keeps those legs it is broken, whatever it does for the pair. That check runs below
and its verdict prints whether or not it is convenient.

KNOWN LIMIT, stated rather than discovered later. The sampler keeps the top 4 processes above
10 % cpu and records no ppid, so a foreign job under 10 %, or below four busier ones, is
invisible, and a bare `multiprocessing.spawn` child cannot be attributed by ancestry after the
fact. Those are counted and reported separately as ambiguous rather than folded into either side.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

MINE = ("fold_ab_flip.py", "sample_contention.py", "narrowq_pair.sh",
        "/home/ttuser/.coworker/wt/land-standing")
INFRA = ("stallwatch.py", "wdtrace3.py", "qbcard-bisect", "qbfix")
AMBIG = ("multiprocessing.spawn",)


def classify(cmd):
    if any(s in cmd for s in INFRA):
        return "infra"
    if any(s in cmd for s in MINE):
        return "mine"
    if any(s in cmd for s in AMBIG):
        return "ambiguous"
    return "foreign"


def score_leg(fold, rows):
    ins = [r for r in rows if fold["t_start"] <= r["t"] <= fold["t_end"]]
    worst_f, worst_a, who = 0.0, 0.0, ""
    for r in ins:
        f = a = 0.0
        names = []
        for e in (r.get("top") or []):
            c = classify(e.get("cmd", ""))
            if c == "foreign":
                f += e["pcpu"]
                names.append("%s@%.0f%%" % (e["cmd"].split()[0].split("/")[-1], e["pcpu"]))
            elif c == "ambiguous":
                a += e["pcpu"]
        if f > worst_f:
            worst_f, who = f, ", ".join(names)
        worst_a = max(worst_a, a)
    lo = [r["loadavg"][0] for r in ins]
    return {"n": len(ins), "foreign": worst_f, "ambig": worst_a, "who": who,
            "load_max": max(lo) if lo else None}


def main():
    man = json.load(open(ROOT / "perf/land_standing/narrowq_bank_sources.json"))
    print("narrow-q legs scored on FOREIGN cpu (contention) vs raw loadavg1 (which includes the "
          "fold itself)\n")
    print("  %-30s %-3s %-5s %8s %9s %9s  %s" % (
        "source", "arm", "rep", "runtime", "load_max", "foreign%", "worst foreign process"))
    legs = []
    for src in man["sources"]:
        art_p, tr_p = ROOT / src["artifact"], ROOT / src["trace"]
        if not art_p.exists() or not tr_p.exists():
            continue
        rows = [json.loads(l) for l in open(tr_p) if l.strip()]
        rows = [r for r in rows if "t" in r]
        art = json.load(open(art_p))
        for cell in art["cells"]:
            for f in cell.get("folds", []):
                s = score_leg(f, rows)
                nm = src["artifact"].split("/")[-1][:30]
                print("  %-30s %-3s rep%-2d %7.1fs %9.2f %9.1f  %s" % (
                    nm, f["arm"], f["rep"], f["runtime_s"], s["load_max"] or -1,
                    s["foreign"], s["who"] or "-"))
                legs.append((nm, f, s))
    print("\nambiguous (unattributable multiprocessing.spawn) worst per leg:")
    for nm, f, s in legs:
        print("  %-30s %-3s rep%-2d  %7.1f%%" % (nm, f["arm"], f["rep"], s["ambig"]))

    print("\n--- NEGATIVE CONTROL: the retake's reps 2-4 had a real co-tenant from 22:11:40Z ---")
    bad = [(nm, f, s) for nm, f, s in legs if "retake" in nm and f["rep"] >= 2]
    good = [(nm, f, s) for nm, f, s in legs if not ("retake" in nm and f["rep"] >= 2)]
    print("  contended legs (must stay cut): foreign%% = %s" %
          ", ".join("%.0f" % s["foreign"] for _, _, s in bad))
    print("  clean legs    (should survive): foreign%% = %s" %
          ", ".join("%.0f" % s["foreign"] for _, _, s in good))
    if bad and good and min(s["foreign"] for _, _, s in bad) > max(s["foreign"] for _, _, s in good):
        print("  DISCRIMINATES: every contended leg carries more foreign cpu than every clean one.")
    else:
        print("  DOES NOT DISCRIMINATE on this data -- the bar is not usable as written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
