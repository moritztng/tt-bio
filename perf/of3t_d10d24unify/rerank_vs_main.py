"""Price the unified ranking rule against MAIN, which nobody had done.

`of3t-rankunify` built and measured the rule against `wk/of3t`, where OpenFold3 already carried
its no-interface repair. Its tables therefore report two models moving, and Moritz's 9629
decision quotes that. `origin/main` is what users get, and main never took the repair: its
OpenFold3 still computes `0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*clash` with ipTM identically
zero on a monomer. So the composition-to-main delta covers THREE models, and the third moves in
the direction the decision did not have in front of it.

This does not re-derive the rule or the interface proof; both are `of3t-rankunify`'s and are
reused as they stand. It re-ranks that row's own recorded cells under main's three expressions.
A ranking rule is post-forward, so re-ranking the per-sample scalars a fold recorded gives
exactly the structure another rule would have served. Every scalar and every `rmsd_ca` here is
read off `perf/of3t_rankunify/cells.json`, folded on qb2 p300c at AICLK median 1350 sampled
during the fold. Nothing is re-folded and nothing is estimated.
"""
import json
import math
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
CELLS = HERE.parent / "of3t_rankunify" / "cells.json"

# Real globular monomers only. The peptide and the multi-chain targets have 16-36 A seed
# floors or no ground truth, so they cannot carry an accuracy claim; `of3t-rankunify` states
# the same bound.
TARGETS = {"examples/ubq.yaml": "ubq", "examples/prot.yaml": "prot"}


# --- the four pre-images, transcribed from origin/main ------------------------------------
def main_openfold3(s):
    """openfold3_fold.py:277 on main. AF3 SI 5.9.3, with no answer for ipTM == 0."""
    return (0.8 * (s["iptm"] or 0.0) + 0.2 * s["ptm"] + 0.5 * s["disorder"]
            - 100.0 * s["has_clash"])


def main_rf3(s):
    """rf3/confidence.py:108 + :190 on main. ipTM is None on a monomer and pTM is substituted
    for it, giving 0.8*pTM + 0.2*pTM = 1.0*pTM, which is exactly what the Protenix site
    computes; and the score worker.py ORDERS on is rounded to 4 decimals."""
    iptm = s["iptm"] if s["iptm"] is not None else s["ptm"]
    return round(0.8 * iptm + 0.2 * s["ptm"] - 100 * int(bool(s["has_clash"])), 4)


def main_protenix(s):
    """worker.py:1066 `_protenix_emit._score` on main."""
    if (s["iptm"] or 0.0) > 0.0:
        return 0.8 * s["iptm"] + 0.2 * s["ptm"]
    return s["ptm"] if s["ptm"] > 0.0 else s["plddt"]


def unified(s):
    """tt_bio/ranking.py. A site that computes no disorder or clash term passes 0.0."""
    iptm = s["iptm"]
    interface = 0.8 * float(iptm) if iptm else 0.8 * s["plddt"]
    return interface + 0.2 * s["ptm"] + 0.5 * s["disorder"] - 100.0 * s["has_clash"]


def unified_at(model):
    """The unified rule as each site actually calls it: rf3 and protenix pass 0.0 for the
    terms they do not compute, so the arm below is what the shipped code does, not the rule
    in the abstract."""
    if model in ("rf3",):
        return lambda s: unified({**s, "disorder": 0.0})
    if model in ("protenix-v2", "opendde"):
        return lambda s: unified({**s, "disorder": 0.0, "has_clash": 0.0})
    return unified


MAIN = {"openfold3": main_openfold3, "rf3": main_rf3,
        "protenix-v2": main_protenix, "opendde": main_protenix}


def served(samples, rule):
    """Rank 0 under `rule`. Every site sorts descending with a STABLE sort, so a tie is
    broken by sample index -- which is the rf3 rounding defect, reproduced rather than
    idealised away."""
    order = sorted(range(len(samples)), key=lambda i: -rule(samples[i]))
    return order[0]


def sign_test(better, worse):
    """Two-sided exact binomial at p = 0.5."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def main_rf3_unrounded(s):
    """rf3's MAIN rule with the rounding defect removed and nothing else changed. The gap
    between this and `main_rf3` is what the ordering fix is worth on its own; the gap between
    this and the unified arm is what the rule change is worth."""
    iptm = s["iptm"] if s["iptm"] is not None else s["ptm"]
    return 0.8 * iptm + 0.2 * s["ptm"] - 100 * int(bool(s["has_clash"]))


def interface_branch(cells):
    """With an interface, does the unified rule return what main's site returned, to the bit?

    OpenFold3 and Protenix/OpenDDE: yes, and it is checked rather than argued. RF3: the VALUE
    differs, because main rounds the score it orders on to 4 decimals and this branch does
    not. So the check for rf3 is on the ORDER, and a disagreement there is the rounding defect
    firing, not the rule changing.
    """
    print("\nInterface branch, every multi-chain cell on disk:")
    for tag, c in sorted(cells.items()):
        ss = c["samples"]
        if not any(s["iptm"] for s in ss):
            continue
        model = c["model"]
        old, new = MAIN[model], unified_at(model)
        worst = max(abs(new(s) - old(s)) for s in ss)
        same_order = ([s["sample"] for s in sorted(ss, key=lambda s: -old(s))]
                      == [s["sample"] for s in sorted(ss, key=lambda s: -new(s))])
        print("  %-28s n=%d  worst |unified-main| = %.3e  order preserved: %s"
              % (tag, len(ss), worst, same_order))


def main():
    cells = json.loads(CELLS.read_text())
    rows = defaultdict(list)
    for tag, c in cells.items():
        t = TARGETS.get(c["target"])
        if t is None:
            continue
        if not all(s["rmsd_ca"] is not None for s in c["samples"]):
            continue
        if not c.get("rank_map_verified"):
            sys.exit("cell %s is not rank-map verified; it must not be scored" % tag)
        rows[(c["model"], t)].append(c)

    out, tot_b, tot_w, tot_same = [], 0, 0, 0
    for (model, t), cs in sorted(rows.items()):
        old, new = MAIN[model], unified_at(model)
        a = b = 0.0
        better = worse = same = 0
        for c in sorted(cs, key=lambda c: c["seed"]):
            ss = c["samples"]
            ia, ib = served(ss, old), served(ss, new)
            ra, rb = ss[ia]["rmsd_ca"], ss[ib]["rmsd_ca"]
            a += ra
            b += rb
            if ss[ia]["sha256"] == ss[ib]["sha256"]:
                same += 1
            elif rb < ra:
                better += 1
            else:
                worse += 1
        n = len(cs)
        out.append({"model": model, "target": t, "seeds": n,
                    "main_A": a / n, "unified_A": b / n, "delta_A": (b - a) / n,
                    "unchanged": same, "better": better, "worse": worse})
        tot_b += better
        tot_w += worse
        tot_same += same

    # rf3 only: the ordering fix and the rule change, priced apart.
    print("rf3, the two changes priced apart (mean served Ca-RMSD):")
    for t in ("ubq", "prot"):
        cs = sorted(rows[("rf3", t)], key=lambda c: c["seed"])
        arms = [("main (rounded order, 1.0*pTM)", main_rf3),
                ("ordering fix only", main_rf3_unrounded),
                ("ordering fix + unified rule", unified_at("rf3"))]
        vals = [sum(c["samples"][served(c["samples"], r)]["rmsd_ca"] for c in cs) / len(cs)
                for _, r in arms]
        print("  %-5s %s" % (t, "  ".join("%s %.4f" % (n, v)
                                          for (n, _), v in zip(arms, vals))))
    print()

    p = sign_test(tot_b, tot_w)
    print("%-12s %-5s %5s %9s %9s %9s  %s" % (
        "model", "targ", "seeds", "main A", "unified A", "delta A", "unch/bett/worse"))
    for r in out:
        print("%-12s %-5s %5d %9.4f %9.4f %+9.4f  %d/%d/%d" % (
            r["model"], r["target"], r["seeds"], r["main_A"], r["unified_A"], r["delta_A"],
            r["unchanged"], r["better"], r["worse"]))
    print("\n%d folds: %d unchanged, %d better, %d worse. Two-sided sign test p = %.4f"
          % (tot_same + tot_b + tot_w, tot_same, tot_b, tot_w, p))
    (HERE / "rerank_vs_main.json").write_text(json.dumps(
        {"rows": out, "unchanged": tot_same, "better": tot_b, "worse": tot_w,
         "sign_test_p": p, "source": "perf/of3t_rankunify/cells.json"},
        indent=1, sort_keys=True) + "\n")
    interface_branch(cells)


if __name__ == "__main__":
    main()
