#!/usr/bin/env python3
"""Does the wheel's fused softmax backward clear the accuracy bar at bf16?

Prints `--moreh` when it does and nothing when it does not, so the A/B either takes the arm or
runs the precision stack alone. The default on any doubt is to DROP the arm: an unreadable
artifact, a missing arm, a refusal recorded as an error, or a rel L2 past the bar all mean the
same thing here, which is that nothing has shown this kernel to be right on Blackhole. Its
sibling `moreh_layer_norm_backward` is wrong there at dx 2.741e+06 rel L2 in bf16 (upstream
#12349), so "the op exists in the wheel and ran" is not evidence about this op either.

The bar is rel L2 against a float64 reference built from the same operands, never against
another device arm.
"""
import json
import sys

# The renorm-keeping arm is the one the A/B would actually route through; `moreh_bf16` drops the
# renorm, which is default-on since Moritz's ask-9629 ruling, so it cannot carry the arm alone.
REQUIRED = "moreh_rn_bf16"


def verdict(blob, bar):
    rows = blob.get("rows") or []
    seen = []
    for row in rows:
        rec = row.get(REQUIRED) if isinstance(row, dict) else None
        if not isinstance(rec, dict):
            continue
        rel = rec.get("rel_l2_vs_f64")
        if not isinstance(rel, (int, float)):
            return False, f"{REQUIRED} at n={row.get('n')}: no rel_l2_vs_f64 ({rec})"
        seen.append((row.get("n"), rel))
        if rel > bar:
            return False, f"{REQUIRED} at n={row.get('n')}: rel L2 {rel:.3e} > bar {bar:.1e}"
    if not seen:
        return False, f"{REQUIRED} appears in no row -- the arm did not run or was refused"
    worst = max(r for _, r in seen)
    return True, f"{REQUIRED} clears the bar at every n, worst rel L2 {worst:.3e} <= {bar:.1e}"


def main():
    path, bar = sys.argv[1], float(sys.argv[2])
    try:
        blob = json.load(open(path))
    except Exception as exc:
        print(f"no usable probe artifact ({exc}) -- dropping the moreh arm", file=sys.stderr)
        return 1
    ok, why = verdict(blob, bar)
    print(why, file=sys.stderr)
    if ok:
        print("--moreh")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
