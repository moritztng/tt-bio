"""Pool every valid taped step this row measured and ask whether the lever is visible.

A step is VALID here only if it actually trained: 9,888 tape nodes and a non-zero weight
reach. Pass 1's reps 1 and 2 fail that and are excluded by the rule rather than by hand.

The comparison is a two-sided exact permutation test on the difference of medians, because
with four steps a side and an in-process A/A floor of 100.58 s, a t-test's assumptions are
the thing in question. `--within` restricts to reps that shared one process, which is the
only pairing that holds the kernel cache and the weights fixed.
"""
import argparse
import itertools
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def rows(paths):
    for p in paths:
        d = json.loads(p.read_text())
        for r in d.get("reps", []):
            if d["config"].get("taped") is False:
                continue
            if r.get("tape_nodes", 0) < 100 or not r.get("backward_valid"):
                r["excluded"] = "trained nothing: %s nodes, %s" % (
                    r.get("tape_nodes"), r.get("params_with_grad"))
            flag = r.get("renorm_flag")
            if flag is None:                      # pass-1 arms predate --renorm-per-rep
                flag = (d["env"].get("renorm") or "1") != "0"
            yield {"arm": p.stem, "rep": r["rep"], "cold": r["cold"], "on": bool(flag),
                   "step_s": r.get("step_s"), "backward_s": r.get("backward_s"),
                   "nodes": r.get("tape_nodes"), "reach": r.get("params_with_grad"),
                   "excluded": r.get("excluded")}


def perm_p(a, b):
    """Two-sided exact permutation p on |median(a) - median(b)|."""
    obs = abs(statistics.median(a) - statistics.median(b))
    pool, n, hits, total = list(a) + list(b), len(a), 0, 0
    for idx in itertools.combinations(range(len(pool)), n):
        left = [pool[i] for i in idx]
        right = [pool[i] for i in range(len(pool)) if i not in idx]
        total += 1
        if abs(statistics.median(left) - statistics.median(right)) >= obs - 1e-9:
            hits += 1
    return obs, hits / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="step_*.json")
    ap.add_argument("--within", default="", help="one arm stem, for the in-process pairing")
    ap.add_argument("--steady-only", action="store_true")
    a = ap.parse_args()
    rs = [r for r in rows(sorted(OUT.glob(a.glob)))]
    print(f"{'arm':<22}{'rep':>4}{'lever':>7}{'cold':>6}{'step_s':>10}{'bw_s':>10}"
          f"{'nodes':>8}  reach")
    for r in rs:
        print(f"{r['arm']:<22}{r['rep']:>4}{'ON' if r['on'] else 'OFF':>7}"
              f"{'yes' if r['cold'] else '':>6}{r['step_s'] or 0:>10.2f}"
              f"{r['backward_s']:>10.2f}{r['nodes']:>8}  {r['reach']}"
              + ("   EXCLUDED: " + r["excluded"] if r["excluded"] else ""))
    use = [r for r in rs if not r["excluded"]]
    if a.within:
        use = [r for r in use if r["arm"] == a.within]
    if a.steady_only:
        use = [r for r in use if not r["cold"]]
    on = [r["step_s"] for r in use if r["on"]]
    off = [r["step_s"] for r in use if not r["on"]]
    print(f"\nvalid steps: {len(use)}   ON {len(on)}   OFF {len(off)}"
          f"{'   within ' + a.within if a.within else ''}"
          f"{'   steady only' if a.steady_only else ''}")
    for name, xs in (("ON", on), ("OFF", off)):
        if xs:
            print(f"  {name:<4} median {statistics.median(xs):8.2f} s   min {min(xs):8.2f}"
                  f"   max {max(xs):8.2f}   spread {max(xs) - min(xs):7.2f}")
    if len(on) >= 2 and len(off) >= 2:
        obs, p, total = perm_p(on, off)
        print(f"\n  |median(ON) - median(OFF)| = {obs:.2f} s, exact permutation p = {p:.4f} "
              f"over {total} splits")
        print(f"  the A/A floor to beat is the widest same-lever pair: "
              f"ON {max(on) - min(on):.2f} s, OFF {max(off) - min(off):.2f} s")
    elif on and off:
        print(f"\n  |median(ON) - median(OFF)| = "
              f"{abs(statistics.median(on) - statistics.median(off)):.2f} s, and with "
              f"{len(on)}/{len(off)} steps no permutation test is worth running")


if __name__ == "__main__":
    main()
