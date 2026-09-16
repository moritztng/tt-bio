#!/usr/bin/env python3
"""CPU-only accounting audit. No torch/ttnn imports, device opens, or timing estimates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
from itertools import product, zip_longest
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf/roof_budget"))
import exec_flops as legacy  # noqa: E402 (stdlib-only graph reader)

CONVENTION = "dense contractions: 2 FLOPs per multiply-accumulate (including initial add)"
# These labels exclude floating arithmetic under this convention, not instructions or traffic.
NON_ARITHMETIC = legacy.FREE - {"ttnn.typecast"} | {
    "ttnn.pad", "ttnn.from_device", "ttnn.to_torch", "ttnn.from_torch",
}
SYMBOLIC = {
    "ttnn.layer_norm": "reductions, centering, variance, epsilon, rsqrt, affine; schedule unknown",
    "ttnn.softmax": "max reduction, subtract, exp, sum reduction, reciprocal, multiply",
    "ttnn.cos": "cos evaluations; no FLOP coefficient assigned",
    "ttnn.typecast": "numeric conversions; separate from add/multiply FLOPs",
    "ttnn.transformer.scaled_dot_product_attention":
        "QK^T and PV plus scale/mask/softmax; roles, causal/chunk schedule and lowering needed",
}
SOURCES = [
    "perf/roof_budget/exec_flops.py", "perf/b2x_difflayer/itemize.py",
    "perf/roof_budget/roof_budget_table.py", "tt_bio/tenstorrent.py",
    "tt_bio/mm_generic.py", "tt_bio/triatt_qkv.py", "tt_bio/trimul_tail.py", "tt_bio/token_axis.py",
    "tt_bio/sdpa_generic.py", "tt_bio/triatt_sdpa.py", "tt_bio/softmax_generic.py",
    "tt_bio/mm_dualnoc.py", "tt_bio/reblock_permute.py",
    "tt_bio/kernels/reblock_permute/compute_reblock_permute.cpp",
    "tt_bio/kernels/reblock_permute_gated/compute_reblock_permute_gated.cpp",
    "tt_bio/kernels/trimul_tail/compute.cpp",
    "tt_bio/kernels/triatt_sdpa/compute/sdpa.cpp",
    "tt_bio/kernels/triatt_sdpa/compute/compute_common.hpp",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape(value):
    value = tuple(value)
    if len(value) < 2 or any(type(d) is not int or d <= 0 for d in value):
        raise ValueError("requires positive, rank >= 2 integer shapes")
    return value


def broadcast(a, b):
    dims = []
    for x, y in zip_longest(reversed(a), reversed(b), fillvalue=1):
        if x != y and x != 1 and y != 1:
            raise ValueError("incompatible batch broadcast")
        dims.append(max(x, y))
    return tuple(reversed(dims))


def matrix_work(a, b, out=None, *, transpose_a=False, transpose_b=False):
    """Exact shape arithmetic; tile32 is a convention, NOT an issued-instruction census."""
    a, b = shape(a), shape(b)
    if transpose_a:
        a = a[:-2] + (a[-1], a[-2])
    if transpose_b:
        b = b[:-2] + (b[-1], b[-2])
    m, k, kb, n = *a[-2:], *b[-2:]
    if k != kb:
        raise ValueError("contraction K mismatch")
    batch = broadcast(a[:-2], b[:-2])
    expected = batch + (m, n)
    if out is not None and shape(out) != expected:
        raise ValueError("output shape mismatch")
    mp, np = legacy.pad2((m, n))
    kp = ((k + 31) // 32) * 32
    return {
        "batch_shape": list(batch), "M": m, "N": n, "K": k,
        "output_shape": list(expected),
        "shape_flops": 2 * prod(batch) * m * n * k,
        "tile32_flops": 2 * prod(batch) * mp * np * kp,
    }


def known_fused_matrix(kind, pairs):
    """Matrix components for SOURCE-IDENTIFIED kernels, never selected by tensor rank.

    Each pair is the effective activation/weight matrix shape after any documented
    head-major reinterpretation. Split output widths stay in the weight's full N.
    These are formula controls, not annotations retrofitted onto opaque captures.
    """
    arity = {"mm_generic": 1, "triatt_qkv": 1, "trimul_tail": 2}
    if kind not in arity or len(pairs) != arity[kind]:
        raise ValueError("unsupported identity or number of contractions")
    if any(len(a) != 2 or len(b) != 2 for a, b in pairs):
        raise ValueError("supply effective 2D matrices, including head-major reinterpretation")
    terms = [matrix_work(a, b) for a, b in pairs]
    if kind == "trimul_tail" and terms[0] != terms[1]:
        raise ValueError("trimul_tail requires identical activation and weight shapes")
    return {
        "matrix_shape_flops": sum(t["shape_flops"] for t in terms),
        "matrix_tile32_flops": sum(t["tile32_flops"] for t in terms),
        "remaining": ["sigmoid and gate multiply"] if kind == "trimul_tail" else [],
        "issued_flops": None,
    }


def dense_attention_matrix(q, k, v):
    """Known noncausal dense QK^T + PV only; no inferred identity or SFPU coefficient."""
    qk = matrix_work(q, k, transpose_b=True)
    pv = matrix_work(qk["output_shape"], v)
    return {"matrix_shape_flops": qk["shape_flops"] + pv["shape_flops"],
            "matrix_tile32_flops": qk["tile32_flops"] + pv["tile32_flops"],
            "remaining": ["scale, mask, softmax and streaming recurrence"],
            "issued_flops": None}


def matrix_candidates(device, by_id, outs):
    """Use ordered device operands, not a deduplicated bag containing outputs/biases.

    Historical MatmulParams has positional, partly opaque attributes. Enumerate both
    transpose flags and accept only if every shape-compatible interpretation agrees.
    This proves the contraction's count, not its epilogue or instruction schedule.
    """
    ids = device.get("input_tensors", [])
    if len(ids) < 2:
        raise ValueError("missing ordered MatmulDeviceOperation inputs")
    a, b = [legacy._shape((by_id.get(i, {}).get("params") or {}).get("shape"))
            for i in ids[:2]]
    if a is None or b is None:
        raise ValueError("missing ordered operand shapes")
    candidates = []
    for out, ta, tb in product(set(outs), (False, True), (False, True)):
        try:
            candidates.append(matrix_work(a, b, out, transpose_a=ta, transpose_b=tb))
        except ValueError:
            pass
    counts = {(t["shape_flops"], t["tile32_flops"]) for t in candidates}
    if len(counts) != 1:
        raise ValueError("no unique matrix count from ordered inputs and outputs")
    return {"shape_flops": candidates[0]["shape_flops"],
            "tile32_flops": candidates[0]["tile32_flops"],
            "candidates": candidates, "ordered_input_shapes": [a, b]}


def classify(name, ins, outs, internal, by_id):
    row = {"status": "uncounted", "matrix": None, "elementwise": None,
           "remaining": [], "issued_flops": None, "machine_instructions": None}
    if name in NON_ARITHMETIC:
        row.update(status="exact_convention", non_arithmetic=True,
                   convention="0 floating add/multiply FLOPs; movement/metadata excluded")
        children = {(n.get("params") or {}).get("name", "") for n in internal[1:]}
        if any(c.startswith("ttnn.") and c not in NON_ARITHMETIC for c in children):
            row.update(status="uncounted", non_arithmetic=False,
                       remaining=["arithmetic child inside nominally non-arithmetic parent"])
    elif name in legacy.MATMUL:
        devs = [n for n in internal if (n.get("params") or {}).get("name")
                == "MatmulDeviceOperation"]
        try:
            if len(devs) != 1:
                raise ValueError("requires exactly one MatmulDeviceOperation")
            row["matrix"] = matrix_candidates(devs[0], by_id, outs)
            row["status"] = "exact_convention"
            row["convention"] = CONVENTION + "; matrix component only"
            row["remaining"] = ["bias/fused activation attributes and program schedule"]
        except (ValueError, TypeError) as exc:
            row["remaining"] = [str(exc)]
    elif name == "ttnn.generic_op":
        row["remaining"] = ["opaque kernel identity, operand roles, defines, runtime/compile args"]
    elif name in SYMBOLIC or name in {"ttnn.add", "ttnn.add_", "ttnn.multiply", "ttnn.multiply_"}:
        row["status"] = "symbolic"
        row["elementwise"] = {
            "output_shapes": outs,
            "output_elements": [prod(s) for s in outs],
            "expression": SYMBOLIC.get(name, "one add/multiply per result plus any fused activation"),
        }
        row["remaining"] = ["resolved output/alias roles, fusion attributes and reduction/SFPU lowering"]
    else:
        row["remaining"] = ["unsupported operation class"]
    return row


def audit_capture(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        data = json.load(stream)
    nodes = data["nodes"] if isinstance(data, dict) and "nodes" in data else data
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("capture must contain a nonempty node list")
    by_id = {n["counter"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate node counters")
    ops, owner = legacy.top_level_spans(nodes)
    _, ins, outs = legacy.operands(nodes)
    internal = defaultdict(list)
    for n in nodes:
        if n["node_type"] == "function_start" and n["counter"] in owner:
            internal[owner[n["counter"]]].append(n)
    starts = sum(n["node_type"] == "function_start" for n in nodes)
    ends = sum(n["node_type"] == "function_end" for n in nodes)
    # Equal totals alone do not prove a well-formed stack. Check prefix balance too.
    stack = []
    balanced = True
    for n in nodes:
        name = (n.get("params") or {}).get("name")
        if n["node_type"] == "function_start":
            stack.append(name)
        elif n["node_type"] == "function_end":
            balanced &= bool(stack) and stack[-1] == name
            if stack:
                stack.pop()
    balanced &= not stack
    rows = []
    classes = {}
    for i, op in enumerate(ops):
        row = classify(op["name"], ins[i], outs[i], internal[i], by_id)
        row.update(counter=op["start"], name=op["name"])
        lf, pf, kind = legacy.op_flops(op["name"], ins[i], outs[i])
        row["legacy"] = {"logical": lf, "padded": pf, "kind": kind}
        if row["matrix"]:
            row["legacy_matrix_agrees"] = (lf, pf) == (
                row["matrix"]["shape_flops"], row["matrix"]["tile32_flops"])
        rows.append(row)
        c = classes.setdefault(op["name"], {"occurrences": 0, "statuses": Counter(),
            "matrix_shape_subtotal": None, "matrix_tile32_subtotal": None,
            "legacy_matrix_disagreements": 0, "remaining": set()})
        c["occurrences"] += 1
        c["statuses"][row["status"]] += 1
        c["remaining"].update(row["remaining"])
        if row["matrix"]:
            c["matrix_shape_subtotal"] = (c["matrix_shape_subtotal"] or 0) + row["matrix"]["shape_flops"]
            c["matrix_tile32_subtotal"] = (c["matrix_tile32_subtotal"] or 0) + row["matrix"]["tile32_flops"]
            c["legacy_matrix_disagreements"] += not row["legacy_matrix_agrees"]
    for c in classes.values():
        c["statuses"] = dict(c["statuses"])
        c["remaining"] = sorted(c["remaining"])
        # Range fallback yields candidate spans, not a certified disjoint partition.
        c["subtotal_scope"] = "within capture only" if balanced else "candidate spans; ownership unverified"
    blockers = sorted({r for row in rows for r in row["remaining"]})
    if not balanced:
        blockers.append("unbalanced/mismatched function spans: legacy ownership is not certified")
    if not ops:
        blockers.append("no captured operations")
    # Counts above describe mathematical work. Issued work additionally needs program schedules.
    if any(not r.get("non_arithmetic") for r in rows):
        blockers.append("physical program padding/skipped tiles/redundant work not captured")
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": digest(path), "function_starts": starts, "function_ends": ends,
        "ownership": "balanced_stack" if balanced else "unverified_legacy_spans",
        "legacy_parser": "stack" if starts == ends else "range",
        "operations": len(rows), "classes": dict(sorted(classes.items())), "rows": rows,
        "exact_total_allowed": not blockers, "exact_total_flops": 0 if not blockers else None,
        "blockers": blockers, "cycles": None,
    }


def legacy_controls():
    """Independent known answers exposing accepted guesses. Failure => STOP, not an audit crash."""
    cases = [
        ("rectangular_control", "ttnn.matmul",
         [(None, (64, 32)), (1, (32, 16))], [(64, 16)], 65536),
        ("square_M_equals_K_rhs_mistaken_for_activation", "ttnn.matmul",
         [(None, (32, 32)), (1, (32, 64))], [(32, 64)], 131072),
        ("broadcast_batch_missing_lhs_rows", "ttnn.matmul",
         [(None, (1, 3, 5)), (1, (7, 5, 11))], [(7, 3, 11)], 2310),
        ("generic_rank2_output_is_not_another_weight", "ttnn.generic_op",
         [(None, (32, 32)), (1, (32, 32)), (2, (32, 32))], [], 65536),
        ("generic_sdpa_batch_one_is_not_layout", "ttnn.generic_op",
         [(0, (1, 2, 32, 64)), (1, (1, 2, 32, 64)), (2, (1, 2, 32, 64)),
          (3, (1, 2, 32, 32)), (4, (1, 2, 32, 64))], [], 524288),
    ]
    result = []
    for name, op, ins, outs, expected in cases:
        observed, padded, kind = legacy.op_flops(op, ins, outs)
        result.append({"name": name, "op": op, "inputs": ins, "outputs": outs,
                       "expected_matrix_shape_flops": expected, "legacy_shape_flops": observed,
                       "legacy_padded": padded, "legacy_kind": kind,
                       "passed": observed == expected})
    return result


def build_report(paths):
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("duplicate capture path; refusing repeat inclusion")
    captures = [audit_capture(p.resolve()) for p in paths]
    inventory = {}
    for cap in captures:
        for name, c in cap["classes"].items():
            row = inventory.setdefault(name, {"capture_occurrences": 0, "statuses": Counter()})
            row["capture_occurrences"] += c["occurrences"]
            row["statuses"].update(c["statuses"])
    for c in inventory.values():
        c["statuses"] = dict(c["statuses"])
    controls = legacy_controls()
    return {
        "schema_version": 1, "measurement": "CPU replay and exact integer count validation only",
        "clock": "CPU only; 1350 MHz target not measured", "cycles": None,
        "conventions": {
            "matrix": CONVENTION,
            "shape": "captured tensor logical shape; includes model-side bucketing, not original input work",
            "tile32": "round effective M,N,K to 32; exact conditional arithmetic, not issued program work",
            "sfpu": "symbolic evaluations only; no exp/sqrt/etc FLOP coefficients",
            "instructions": "not counted; require compiled kernels and dynamic paths",
        },
        "aggregation": "Inventory occurrences only. Captures overlap. No cross-capture FLOP sum or fold total.",
        "source_sha256": {p: digest(ROOT / p) for p in SOURCES},
        "fixture_provenance": json.loads((Path(__file__).parent / "provenance.json").read_text()),
        "captures": captures, "class_inventory": dict(sorted(inventory.items())),
        "legacy_controls": controls,
        "verdict": "STOP" if (any(not c["passed"] for c in controls) or any(
            row.get("legacy_matrix_agrees") is False for cap in captures for row in cap["rows"]
        )) else "GO",
        "exact_whole_table_allowed": False,
        "exact_whole_table_reason": "requires disjoint capture coverage plus complete arithmetic and program contracts",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("captures", nargs="*", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summary", action="store_true", help="omit individual op rows, retain class counts")
    ap.add_argument("--require-exact-total", action="store_true",
                    help="exit 2 if any per-capture total is unresolved, or captures may overlap")
    args = ap.parse_args()
    paths = args.captures or sorted((ROOT / "perf/roof_budget/captures").glob("*.json.gz")) + [
        ROOT / "perf/roof_budget/control_matmul_8192.json",
        ROOT / "perf/c10_flop_contract/control_8192.json"]
    try:
        report = build_report(paths)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        ap.exit(1, f"invalid capture: {exc}\n")
    if args.summary:
        for capture in report["captures"]:
            del capture["rows"]
    data = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(data)
    else:
        print(data, end="")
    # Per-capture arithmetic exactness does not prove that a list is disjoint.
    if args.require_exact_total and (len(paths) != 1 or not report["captures"][0]["exact_total_allowed"]):
        print("REFUSED: exact total lacks complete arithmetic/program/ownership evidence", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
