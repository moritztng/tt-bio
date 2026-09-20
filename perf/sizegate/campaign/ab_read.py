#!/usr/bin/env python3
"""Read a softmax_l1_ab.py artifact and say whether the budget is a lever, inert, or a defect.

Three reads, in this order, because a runtime alone cannot tell them apart:

  defect  the CIF digest moved between budgets. The tail is documented bit-exact under a
          block-size change, so a moving digest is a bug, not a tradeoff -- and it voids the
          runtime comparison too.
  inert   neither l1_blocks nor l1_refused moved between arms: the budget did not change what
          the kernel did at this shape, so any runtime spread is noise by construction.
  lever   the counters moved. Only then is the runtime worth reading, and only against the
          spread between REPS OF THE SAME ARM, which is this instrument's own noise floor.

Usage: ab_read.py <artifact.json> [...]
"""
import json
import statistics
import sys


def summarise(path):
    d = json.load(open(path))
    rows = d["rows"]
    print(f"== {path}  {d['model']}/{d['rung']} card {d['card']}  {len(rows)} folds")
    by = {}
    for r in rows:
        if r.get("rc"):
            print(f"   {r['tag']:16s} rc={r['rc']} FAILED")
            continue
        s, st = r["fp32_softmax"], r["structure"]
        print(f"   {r['tag']:16s} rt={r['runtime_s']:7.1f}s  clk {s and r['aiclk']['min']}"
              f"/{r['aiclk']['median']}/{r['aiclk']['max']} MHz n={r['aiclk']['n']}"
              f"  blocks={s['l1_blocks']} refused={s['l1_refused']} cores={s['l1_cores']}"
              f"  sha={st['sha256']}")
        if not r["tag"].startswith("warmup"):
            by.setdefault(r["budget"], []).append(r)

    if not by:
        print("   no measured arms yet")
        return
    print("   ---")
    digests = {r["structure"]["sha256"] for rs in by.values() for r in rs}
    if len(digests) > 1:
        print(f"   DEFECT: digest moved across budgets {sorted(digests)} -- the tail is"
              " documented bit-exact under a block-size change; the runtimes are void")
        return
    print(f"   digest identical across every arm ({digests.pop()}): bit-exactness holds")

    counters = {b: (rs[0]["fp32_softmax"]["l1_blocks"], rs[0]["fp32_softmax"]["l1_refused"])
                for b, rs in by.items()}
    moved = len(set(counters.values())) > 1
    for b, (bl, rf) in sorted(counters.items()):
        print(f"   {b >> 10:4d} KB  blocks={bl} refused={rf}")
    if not moved:
        print("   INERT at this shape: the counters are identical across budgets, so the"
              " budget changed nothing the kernel did and the runtime spread is noise")

    floor = max((max(x["runtime_s"] for x in rs) - min(x["runtime_s"] for x in rs))
                for rs in by.values() if len(rs) > 1) if any(len(rs) > 1 for rs in by.values()) else None
    means = {b: statistics.fmean(x["runtime_s"] for x in rs) for b, rs in by.items()}
    best, worst = min(means, key=means.get), max(means, key=means.get)
    spread = means[worst] - means[best]
    print(f"   runtime: best {best >> 10} KB {means[best]:.1f}s, worst {worst >> 10} KB"
          f" {means[worst]:.1f}s, spread {spread:.1f}s ({100 * spread / means[worst]:.2f}%)"
          f" over {min(len(rs) for rs in by.values())}+ reps/arm")
    if floor is None:
        print("   NO NOISE FLOOR YET: one rep per arm cannot separate a lever from a fold-to-fold"
              " swing. Not a verdict.")
    elif spread <= floor:
        print(f"   NOT A WIN: the between-budget spread {spread:.1f}s is inside the same-budget"
              f" rep spread {floor:.1f}s")
    else:
        print(f"   candidate: between-budget spread {spread:.1f}s exceeds the same-budget rep"
              f" spread {floor:.1f}s. Needs more reps before it is a number.")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            summarise(p)
        except FileNotFoundError:
            print(f"== {p}: not written yet")
