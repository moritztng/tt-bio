"""Tabulate an afgrad VJP artifact against the bf16 envelope, and diff two artifacts.

The bar is not zero error: the device computes in bf16, so the reference is torch's OWN bf16
error against the same float64, on the same inputs. `ratio` is device rel_l2 over that envelope,
so ~1.0 means indistinguishable from doing the arithmetic in bf16.

The controls are what make a pass mean anything: a permuted cotangent must collapse `cos`, and a
zero seed must return exactly 0.0. Both are printed, and `--check` exits non-zero if any block
is non-finite, any ratio exceeds the bar, or a control fails to fire.

    grade_table.py <artifact.json> [more.json ...] [--check] [--bar 1.30]
"""
import argparse
import json
import math
import sys
from pathlib import Path


def rows(path):
    return json.loads(Path(path).read_text())


def fmt(x, w=10):
    if x is None:
        return "-".rjust(w)
    if isinstance(x, float) and not math.isfinite(x):
        return ("INF" if x > 0 else "-INF").rjust(w)
    return f"{x:.6g}".rjust(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bar", type=float, default=1.30,
                    help="max device/torch-bf16 rel_l2 ratio; Blackhole's published row is 1.13x")
    a = ap.parse_args()

    bad = []
    for path in a.artifacts:
        d = rows(path)
        print(f"\n=== {path}  n={d['n']} seed={d['seed']} ===")
        print(f"{'block':7}{'device dz':>13}{'torch bf16':>13}{'ratio':>10}{'cos':>11}"
              f"{'device dm':>13}{'perm cos':>13}{'zero seed':>11}")
        for r in d["rows"]:
            dz, bf = r["dz"]["rel_l2"], r["dz_torch_bf16"]["rel_l2"]
            ratio = dz / bf if bf else float("inf")
            perm = (r.get("permuted") or {}).get("dz", {}).get("cos")
            # zero_seed_max_abs is {"dz": ..., "dm": ...}; the control passes only if BOTH are 0.
            zs = r.get("zero_seed_max_abs") or {}
            zero = max(zs.values()) if zs else None
            print(f"{r['block']:7}{fmt(dz,13)}{fmt(bf,13)}{fmt(ratio,10)}{fmt(r['dz']['cos'],11)}"
                  f"{fmt(r['dm']['rel_l2'],13)}{fmt(perm,13)}{fmt(zero,11)}")
            if not all(math.isfinite(v) for v in (dz, r["dz"]["cos"], r["dm"]["rel_l2"])):
                bad.append(f"{path}:{r['block']} non-finite")
            elif ratio > a.bar:
                bad.append(f"{path}:{r['block']} ratio {ratio:.2f}x over the {a.bar}x bar")
            if perm is not None and abs(perm) > 0.01:
                bad.append(f"{path}:{r['block']} permuted control did not collapse (cos {perm})")
            if zero is not None and zero != 0.0:
                bad.append(f"{path}:{r['block']} zero seed returned {zero}, not 0.0")

    if bad:
        print("\nFAIL:")
        for b in bad:
            print("  " + b)
    else:
        print("\nevery block finite, inside the bar, and both controls fired where present")
    return 1 if (bad and a.check) else 0


if __name__ == "__main__":
    sys.exit(main())
