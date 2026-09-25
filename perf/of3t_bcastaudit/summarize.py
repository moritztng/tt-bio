#!/usr/bin/env python3
"""Fold the per-process counts of run_audit.sh into one row per model: summarize.py <root> <name>... --out F.json

Bad forms, from PROBE_ENVELOPE.json: D259 is multiply/mul with a bf16 operand against a
broadcast fp32 one (any axis); D258 is a transpose_a matmul with mixed dtypes. Every other
mixed-dtype call is listed too, so the classification can be re-read against the envelope.
"""
import collections
import json
import sys
from pathlib import Path

MUL = {"multiply", "mul", "multiply_", "mul_"}


def _numel(shape):
    n = 1
    for x in shape:
        n *= x
    return n


def bad(h):
    if h["form"] == "transpose_a_mixed":
        return "D258"
    if h["op"] in MUL and h["form"] == "mixed_bcast":
        if _numel(eval(h["shape_a"])) < _numel(eval(h["shape_b"])):
            return "mul_mixed_bcast_unprobed_order"  # broadcast operand first: probe it first
        return "D259" if h["dtype_a"] == "BFLOAT16" else None  # fp32 x * bf16 b is clean
    return None


def main():
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    root = Path(argv[0])
    names = [a for a in argv[1:argv.index("--out")]]
    rows = {}
    for name in names:
        calls, hits = collections.Counter(), collections.Counter()
        procs = 0
        for f in sorted((root / name / "counts").glob("*.json")):
            d = json.loads(f.read_text())
            if not d["calls"]:
                continue
            procs += 1
            calls.update(d["calls"])
            for h in d["hits"]:
                k = tuple((key, h[key]) for key in ("op", "form", "site", "dtype_a", "dtype_b",
                                                    "shape_a", "shape_b"))
                hits[k] += h["calls"]
        hl = [dict(k, calls=n, bad=bad(dict(k))) for k, n in hits.items()]
        badh = [h for h in hl if h["bad"]]
        rows[name] = {
            "processes_with_calls": procs,
            "calls_wrapped": sum(calls.values()), "calls_by_op": dict(calls),
            "bad_form_calls": sum(h["calls"] for h in badh),
            "bad_sites": sorted({f"{h['bad']} {h['op']} {h['site']}" for h in badh}),
            "mixed_calls_by_form": dict(collections.Counter(
                f"{h['op']}:{h['form']}" for h in hl for _ in range(h["calls"]))),
            "mixed_sites": sorted({f"{h['op']}:{h['form']} {h['site']} {h['dtype_a']}x{h['dtype_b']} "
                                   f"{h['shape_a']}x{h['shape_b']}" for h in hl}),
        }
        r = rows[name]
        print(f"{name}: wrapped {r['calls_wrapped']}, bad {r['bad_form_calls']}, "
              f"bad sites {len(r['bad_sites'])}, mixed forms {r['mixed_calls_by_form']}")
        for s in r["bad_sites"]:
            print("   BAD", s)
    out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
