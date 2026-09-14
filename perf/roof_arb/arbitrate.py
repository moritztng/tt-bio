#!/usr/bin/env python3
"""Three byte totals for one 512 aa fold. Which one, and why the other two are what they are.

    2.9449 TB   real_traffic.py on the tip's captures            floor 6.934 s
    3.4050 TB   roof_table.py on b2x-baseline-attrib's captures  floor 8.017 s
    4.0106 TB   the same, re-run after the RANGE span rule       floor 9.443 s

They are not three readings of one fold. Two axes separate them and this file measures both on the
same code path, changing one thing at a time:

  SPAN   `itemize.top_level_spans` STACK vs RANGE. ttnn.graph drops `function_end` for some device
         operations, so a big capture is unbalanced; STACK then never returns to depth zero and
         collapses 428 top-level ops into 4, and a buffer read by ten ops is charged one read.
         RANGE owns every node by the last ttnn.* `function_start` at or before it and recovers all
         428. This is an instrument bug and it is worth 1.18x.
  TIP    two capture sets of two different commits. The older one folds 7168 atoms where the tip
         folds 4480 and predates every byte-deleting lever the campaign landed. This is a real
         change in the fold and it is worth 0.73x.

On top of that, three charging corrections that apply to ALL of them, from `corrected_traffic.py`.
They interact, so they are reported as an ordered ladder and never as independent addends.

Roofs: measured, not asserted -- starved 8192^2 bf16 add 424.7 GB/s and dense bf16 HiFi4
104.93 TFLOP/s, both from `perf/roof_budget/` in one session on qb2 card 2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RB = ROOT / "perf" / "roof_budget"
BA = ROOT / "perf" / "b2x-baseline-attrib"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RB))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import exec_flops as EF                                                       # noqa: E402
import itemize as IT                                                          # noqa: E402
import corrected_traffic as C                                                 # noqa: E402

TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64",
       "DiffusionModule|"]
CALLS = {TOP[0]: 264, TOP[1]: 16, TOP[2]: 200}

LADDER = [("published", dict(l1=False, pre=False, gate=False)),
          ("+L1", dict(l1=True, pre=False, gate=False)),
          ("+L1+PRE", dict(l1=True, pre=True, gate=False)),
          ("+L1+PRE+GATE", dict(l1=True, pre=True, gate=True))]

_AUTO = IT.top_level_spans


def set_span(rule):
    IT.top_level_spans = (lambda nodes, r="x": _AUTO(nodes, rule))


def cap(d, sig):
    return d / ("cap_" + sig.replace("|", "__") + ".json.gz")


def fold_bytes(capdir, kw):
    tot, per = 0.0, {}
    for sig in TOP:
        mb = C.counts({"nodes": C.load_nodes(str(cap(capdir, sig)))}, **kw)["real_MB"]
        per[sig] = round(mb, 3)
        tot += mb * CALLS[sig]
    return tot / 1e6, per                       # TB


def main():
    ctl = json.loads((RB / "instrument_control.json").read_text())
    st = json.loads((RB / "stream_roof2.json").read_text())
    S = max(r["GBps"] for r in st["stream"] if r["op"] == "add" and r["N"] == 8192) * 1e9
    Cr = max(r["TFLOPs"] for r in ctl["rows"] if r["label"].startswith("cube")) * 1e12
    out = {"stream_roof_GBps": round(S / 1e9, 1), "compute_roof_TFLOPs": round(Cr / 1e12, 2),
           "machine_balance_flop_per_byte": round(Cr / S, 1)}

    # ---- 1. the known-answer control ---------------------------------------------------------
    set_span("auto")
    nodes = C.load_nodes(str(RB / "control_matmul_8192.json"))
    exact = (8192 * 8192 * 3) * 2
    kac = {}
    for name, kw in LADDER:
        c = C.counts({"nodes": nodes}, **kw)
        kac[name] = {"MB": round(c["real_MB"], 3),
                     "ratio_to_exact": round(c["real_MB"] * 1e6 / exact, 6),
                     "terminal_MB": round(c["phantom_read_MB"], 3),
                     "ratio_less_terminal": round((c["real_MB"] - c["phantom_read_MB"]) * 1e6
                                                  / exact, 6)}
    # every row of the ladder, checked arithmetically against the recorded counts
    rows = [r for r in ctl["rows"] if "bytes_counter" in r]
    kac["ladder_all_16"] = {
        "n": len(rows),
        "counter_equals_exact_plus_MN2": all(
            r["bytes_counter"] == r["bytes_arith_min"] + r["batch"] * r["M"] * r["N"] * 2
            for r in rows),
        "ratio_min": round(min(r["byte_ratio"] for r in rows), 6),
        "ratio_max": round(max(r["byte_ratio"] for r in rows), 6)}
    kac["exact_MB"] = exact / 1e6
    out["known_answer_case"] = kac

    # ---- 2. the fused known-answer case: the same shape through a pre-allocated destination ---
    # trimul-inproj, M=262144 K=128 N=640, which the ladder also runs as a plain ttnn.matmul.
    nodes = C.load_nodes(str(cap(RB / "captures",
                                 "TriangleMultiplication|1x512x512x128,1x512x512")))
    M, K, N = 262144, 128, 640
    exact2 = (M * K + K * N + M * N) * 2
    fused = {}
    for name, kw in LADDER:
        c = C.counts({"nodes": nodes}, **kw)
        # ops 0..3 are layer_norm, alloc, alloc, the fused projection: everything the matmul costs
        fused[name] = round(sum(c["by_op"][1:4]) / 1e6, 3)
    fused["exact_MB"] = round(exact2 / 1e6, 3)
    fused["ops"] = "allocate + allocate + the fused generic_op that writes both"
    out["known_answer_case_prealloc"] = fused

    # ---- 3. the three totals, each on its own inputs ------------------------------------------
    rec = {}
    for tip, capdir in (("tip_f072ae02f", RB / "captures"), ("baseline_attrib", BA / "captures")):
        for rule in ("stack", "range"):
            set_span(rule)
            tb, per = fold_bytes(capdir, dict(l1=False, pre=False, gate=False))
            rec[f"{tip}/{rule}"] = {"fold_TB": round(tb, 4),
                                    "floor_s_at_stream_roof": round(tb * 1e12 / S, 3),
                                    "MB_per_call": per}
    out["reconciliation"] = rec
    out["reconciliation_note"] = {
        "span_rule_factor": round(rec["baseline_attrib/range"]["fold_TB"]
                                  / rec["baseline_attrib/stack"]["fold_TB"], 4),
        "code_tip_factor": round(rec["tip_f072ae02f/range"]["fold_TB"]
                                 / rec["baseline_attrib/range"]["fold_TB"], 4)}

    # ---- 4. the corrected fold total, itemised ------------------------------------------------
    set_span("auto")
    lad = {}
    prev = None
    for name, kw in LADDER:
        tb, per = fold_bytes(RB / "captures", kw)
        lad[name] = {"fold_TB": round(tb, 4), "floor_s": round(tb * 1e12 / S, 3),
                     "delta_TB": None if prev is None else round(tb - prev, 4),
                     "delta_s": None if prev is None else round((tb - prev) * 1e12 / S, 3),
                     "MB_per_call": per}
        prev = tb
    out["corrected_ladder"] = lad

    # ---- 5. one floor, max(traffic, compute) at the finest granularity the bytes support ------
    # Two compute roofs, because they answer different questions and only one of them is a roof
    # the fold can reach. `roof-shape-honest-roofs` (state/roof-shape-honest-roofs.md, GO,
    # perf/roof_shape/ on wk/roof-shape-honest-roofs) measured the FLOP-weighted harmonic mean
    # fraction of the dense cube that the fold's OWN matmul shapes reach: 18.8 % on Blackhole over
    # 97.1 % of the census FLOPs. The dense-cube rate is an upper bound no op in this fold reaches.
    SHAPE_FRAC = 0.188
    gran = {"shape_honest_fraction": SHAPE_FRAC,
            "shape_honest_source": "roof-shape-honest-roofs, p150a BH, 15 classes, 97.1 % coverage",
            "shape_honest_TFLOPs": round(Cr * SHAPE_FRAC / 1e12, 2)}
    for name, kw in LADDER:
        row = {}
        for rname, R in (("cube", Cr), ("shape_honest", Cr * SHAPE_FRAC)):
            agg = unit = op = 0.0
            detail = {}
            for sig in TOP:
                nodes = EF.nodes_of(str(cap(RB / "captures", sig)))
                c = C.counts({"nodes": nodes}, **kw)
                B = list(c["by_op"])
                F = [padf if kind in ("matmul", "eltwise", "noshape") else 0
                     for _n, _l, padf, kind in EF.per_op(nodes)]
                assert len(F) == len(B), (len(F), len(B))
                B.append(c["unattributed_B"])      # weight reads with no owning op: 0 FLOPs
                F.append(0)
                assert abs(sum(B) - c["real_MB"] * 1e6) < 1, (sum(B), c["real_MB"])
                n = CALLS[sig]
                agg += sum(B) / S * n
                unit += max(sum(F) / R, sum(B) / S) * n
                op += sum(max(f / R, b / S) for f, b in zip(F, B)) * n
                detail[sig] = {"calls": n, "MB": round(sum(B) / 1e6, 1),
                               "GFLOP": round(sum(F) / 1e9, 1),
                               "AI": round(sum(F) / sum(B), 2),
                               "s_unit": round(max(sum(F) / R, sum(B) / S) * n, 3),
                               "s_per_op": round(sum(max(f / R, b / S)
                                                     for f, b in zip(F, B)) * n, 3)}
            row[rname] = {"traffic_only_no_max_s": round(agg, 3),
                          "max_per_unit_s": round(unit, 3), "max_per_op_s": round(op, 3),
                          "units": detail}
        gran[name] = row
    out["floor_by_granularity"] = gran

    # ---- 6. the guard the audit's two exceptions need ----------------------------------------
    # 20 of the 22 sites have exactly one writer. Two do not: `_gated_rowblocked`
    # (tt_bio/tenstorrent.py:5931) writes `a` and `b` once per row block, and `_row_blocked`
    # (tt_bio/esmc.py:735) writes `dst` once per block through a `ttnn.experimental.view`. On such
    # a buffer the PRE rule would charge writers 2..n a full read each. Neither fires here: ESM-C
    # is not in this fold, and every gated call in the three captures takes the WHOLE wide
    # projection, [1, 512, 512, 512] with the row axis at full length, not a row block.
    set_span("auto")
    guard = {}
    for sig in TOP:
        nodes = C.load_nodes(str(cap(RB / "captures", sig)))
        ops, rows = IT.itemize({"nodes": nodes})
        pa = [r for r in rows if r["kind"] == "DRAM" and r["alloc_op_i"] is not None
              and ops[r["alloc_op_i"]]["name"] == C.ALLOC]
        rb = whole = 0
        for args in C.generic_op_args(nodes).values():
            if len(args) != 3:
                continue
            (sa, _), (sb, zb), (sc, zc) = args
            if (sb, zb) != (sc, zc) or len(sa) != 4 or len(sb) != 4:
                continue
            Cs, N = sb[1], sb[2]
            if not (sb[3] == N and Cs and sa[2] == N and sa[3] % Cs == 0 and sa[3] > 2 * Cs):
                continue
            if sa[1] == N:
                whole += 1
            else:
                rb += 1
        guard[sig] = {"n_prealloc_dram_buffers": len(pa),
                      "prealloc_MB": round(sum(r["size"] for r in pa) / 1e6, 3),
                      "gated_calls_whole_tensor": whole, "gated_calls_row_blocked": rb}
    out["multiwriter_guard"] = guard
    out["multiwriter_guard_clean"] = all(g["gated_calls_row_blocked"] == 0
                                         for g in guard.values())

    (HERE / "arbitration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
