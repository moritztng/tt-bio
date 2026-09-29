#!/usr/bin/env python3
"""Device kernel time per SOURCE LINE, by joining the dispatch order of both instruments.

`census.py` ranks by the ops report's own `OP CODE` + `INPUT_*_PAD[LOGICAL]` key. That key is a
hypothesis about a shape, not a reading of one, and on this model it is wrong for most of the
eltwise class: 5 211 rows of one warm step carry the label `2x1x256x256` while their kernel
durations run from 3 us to 460 us, a spread no single fp32 eltwise shape can produce. The label
cannot be repaired from inside the report, so this joins it to an instrument that does know the
shape -- `call_census.py`'s ordered host trace, which records the operand shapes and the issuing
`tt_bio` frame of every `ttnn` call in dispatch order.

The join is positional and checked, not assumed. Walking both sequences together, a device row is
consumed by the pending host call when its op code is one the call can produce; otherwise it is a
CHILD program the call dispatched implicitly (a `FillPad` before a permute, a `Transpose` inside a
matmul) and is charged to that call without advancing it. A host call that dispatches nothing --
a view-only reshape -- is skipped by a bounded lookahead. The alignment reports its own residual,
and a run where the residual is not ~0 is refused rather than reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

DUR = "DEVICE KERNEL DURATION [ns]"

#: Which device op codes a wrapped `ttnn` call is allowed to BE. Anything else it dispatches is
#: charged to it as a child.
PRODUCES = {
    "add": {"BinaryNgDeviceOperation"}, "subtract": {"BinaryNgDeviceOperation"},
    "multiply": {"BinaryNgDeviceOperation"}, "divide": {"BinaryNgDeviceOperation"},
    "minimum": {"BinaryNgDeviceOperation"}, "maximum": {"BinaryNgDeviceOperation"},
    "lt": {"BinaryNgDeviceOperation"}, "div": {"BinaryNgDeviceOperation"},
    "matmul": {"MatmulDeviceOperation"}, "linear": {"MatmulDeviceOperation"},
    "sum": {"ReduceDeviceOperation"}, "mean": {"ReduceDeviceOperation"},
    "reshape": {"ReshapeViewDeviceOperation"},
    "permute": {"PermuteDeviceOperation", "TransposeDeviceOperation"},
    "transpose": {"TransposeDeviceOperation", "PermuteDeviceOperation"},
    "slice": {"SliceDeviceOperation"}, "concat": {"ConcatDeviceOperation"},
    "softmax": {"SoftmaxDeviceOperation"}, "layer_norm": {"LayerNormDeviceOperation"},
    "relu": {"UnaryDeviceOperation"}, "sqrt": {"UnaryDeviceOperation"},
    "rsqrt": {"UnaryDeviceOperation"}, "reciprocal": {"UnaryDeviceOperation"},
    "neg": {"UnaryDeviceOperation"}, "sigmoid": {"UnaryDeviceOperation"},
    "softplus": {"UnaryDeviceOperation"}, "gtz": {"UnaryDeviceOperation"},
    "typecast": {"UnaryDeviceOperation"}, "clone": {"UnaryDeviceOperation"},
    "zeros": {"FullDeviceOperation"}, "ones_like": {"FullLikeDeviceOperation"},
    "pad": {"PadDeviceOperation", "ConcatDeviceOperation"},
    "to_layout": {"ToLayoutDeviceOperation", "TilizeDeviceOperation", "UntilizeDeviceOperation"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help="ops_perf_results_*.csv")
    ap.add_argument("trace", type=Path, help="call_census.py --json output, with its trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from census import split_last_step

    rows = [r for r in csv.DictReader(args.report.open(newline=""))
            if (r.get(DUR) or "").strip() not in ("", "0")]
    trace = [tuple(t) for t in json.loads(args.trace.read_text())["trace"]]
    # The report holds the cold warmup step and the warm one; the host trace holds only the warm
    # step, so the device side is split by the same rule  uses.
    dev, how = split_last_step(rows)
    if not dev:
        print(how)
        return 1
    print(f"SPLIT: {how}")

    charged = defaultdict(lambda: [0.0, 0])          # (op, shapes, site, kind) -> [ns, n]
    pending: list = []                               # child programs seen since the last self row
    i, child, skipped, orphan = 0, 0, 0, 0
    unknown: set = set()
    for r in dev:
        code, ns = r["OP CODE"], float(r[DUR])
        hit = None
        for d in range(args.lookahead):
            if i + d >= len(trace):
                break
            if code in PRODUCES.get(trace[i + d][0].split(".")[-1], ()):
                hit = d
                break
        if hit is None:
            # A program no host call claims: an implicit child (a `FillPad` a reduction dispatches
            # to zero its operand's tile padding, a `Transpose` inside a matmul). It belongs to the
            # call that is RUNNING, and a child is dispatched BEFORE its parent's own program, so
            # it is held until the next self row identifies the parent rather than charged to
            # whichever call happens to be pending -- a view-only reshape dispatches nothing and
            # would otherwise collect the next call's children.
            pending.append(ns)
            child += 1
            continue
        skipped += hit
        i += hit
        op, shapes, site = trace[i]
        charged[(op, shapes, site, "self")][0] += ns
        charged[(op, shapes, site, "self")][1] += 1
        for cns in pending:
            charged[(op, shapes, site, "child")][0] += cns
            charged[(op, shapes, site, "child")][1] += 1
        pending.clear()
        i += 1
    orphan = len(pending)

    total = sum(v[0] for v in charged.values())
    print(f"JOIN: {len(dev)} device programs against {len(trace)} host calls; "
          f"{child} charged as child programs, {skipped} host calls dispatched nothing, "
          f"{orphan} child programs never claimed by a later call, "
          f"{len(trace) - i} host calls unused")
    if unknown:
        print(f"UNMAPPED host ops: {sorted(unknown)}")
    if orphan > len(dev) * 0.01:
        print("REFUSING: the two sequences did not align")
        return 1
    print(f"DEVICE KERNEL TIME: {total / 1e6:.3f} ms over the joined step")
    print(f"\n{'ms':>9} {'%':>6} {'n':>6} {'us/call':>8}  {'op':<11} {'shapes':<30} site")
    for (op, shapes, site, kind), (ns, n) in sorted(charged.items(), key=lambda kv: -kv[1][0])[:args.top]:
        tag = op if kind == "self" else f"[{op}]"
        print(f"{ns/1e6:>9.3f} {ns/total*100:>6.2f} {n:>6} {ns/n/1e3:>8.1f}  {tag:<11} "
              f"{shapes[:30]:<30} {site}")

    by_site = defaultdict(lambda: [0.0, 0])
    for (op, _s, site, kind), (ns, n) in charged.items():
        by_site[(op if kind == "self" else f"[{op}]", site)][0] += ns
        by_site[(op if kind == "self" else f"[{op}]", site)][1] += n
    print(f"\nBY SITE\n{'ms':>9} {'%':>6} {'n':>6}  {'op':<11} site")
    for (op, site), (ns, n) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:args.top]:
        print(f"{ns/1e6:>9.3f} {ns/total*100:>6.2f} {n:>6}  {op:<11} {site}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "report": str(args.report), "device_programs": len(dev), "host_calls": len(trace),
            "child_programs": child, "empty_host_calls": skipped, "orphans": orphan,
            "device_kernel_ms": total / 1e6,
            "entries": [{"op": o, "kind": k, "shapes": s, "site": si, "ms": v[0] / 1e6, "n": v[1]}
                        for (o, s, si, k), v in
                        sorted(charged.items(), key=lambda kv: -kv[1][0])],
        }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
