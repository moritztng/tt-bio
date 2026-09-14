#!/usr/bin/env python3
"""Print `op_trace.py --score` records from two arms side by side.

Each scored call carries the L2 of every operand and of the output, plus the call's own error
against a float64 reference built from the operands the device got. Reading the two together is
what separates "this op amplified a difference it was handed" from "this op is the one that is
wrong": if the operands agree and the outputs do not, the op is the culprit.

    report_scored.py <A.jsonl> <B.jsonl>
"""
import json
import sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()][1:]


def main() -> int:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    for x, y in zip(a, b):
        if "vs_float64" not in x and "in_l2" not in x:
            continue
        print(f"--- call {x['i']}  {x['op']}  in={[d['shape'] for d in x['in']]} -> {x['out_shape']}")
        print(f"    in_l2   A {x.get('in_l2')}\n            B {y.get('in_l2')}")
        if "mask_l2" in x:
            print(f"    mask_l2 A {x['mask_l2']}  B {y.get('mask_l2')}")
        print(f"    out l2  A {x.get('l2')}  B {y.get('l2')}")
        for tag, r in (("A", x), ("B", y)):
            f = r.get("vs_float64")
            if not f:
                continue
            if "raised" in f:
                print(f"    {tag} vs_f64 RAISED {f['raised']}")
                continue
            print(f"    {tag} vs_f64 rel_l2 {f['rel_l2']:.6e}  max_abs {f['max_abs']:.6e} "
                  f"ref_absmax {f['ref_absmax']:.6f} pcc {f['pcc']:.9f}")
        f = x.get("vs_float64") or {}
        if "kw" in f:
            print(f"    kw {f['kw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
