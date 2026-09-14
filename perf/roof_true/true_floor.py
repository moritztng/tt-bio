#!/usr/bin/env python3
"""max(traffic, arithmetic) PER OP, summed: the 512 aa Boltz-2 fold's one true floor.

A roofline floor is a sum of per-unit maxima and the campaign never took the max. The floor of
record is an arithmetic aggregate (219.49 TFLOP at a FLOP-weighted harmonic 18.8 % of the dense
cube, 11.134 s) with a traffic aggregate beside it (2.8589 TB at 424.7 GB/s, 6.732 s). Neither is
a roofline. Per op some work is traffic bound and some is arithmetic bound, `max(sum) <= sum(max)`,
and the true floor is above both.

No device and no new measurement. Four committed instruments, joined on two keys both sides
already carry:

  op list and parent/child ownership   perf/roof_residual/split_units.py       key: capture + op index
  bytes, all three corrections on      perf/roof_arb/corrected_traffic.py      key: capture + op index
  FLOPs and op kind                    perf/roof_budget/exec_flops.py          key: capture + op index
  shape-honest matmul rates            perf/roof_shape/shape_roofs_pc_bh.json  key: (batch, M, K, N)

The first three all run `itemize()` over the same capture, so "capture + op index" is one key and
not a correspondence anyone invented. The fourth joins on the census shape tuple that
`perf/roof_budget/shape_census.py` builds, which is the key `perf/roof_shape/weigh.py` already
uses; this file takes it per op instead of per fold.

Per top-level ttnn op:

    t_traffic = corrected bytes / 424.7 GB/s
    t_arith   = sum over the op's census shape rows of FLOPs_row / rate(that row's class)
    t         = max(t_traffic, t_arith)

THE OP SET. The three top-level captures are disjoint and tile the fold: 264 PairformerLayer +
16 MSALayer + 200 DiffusionModule is the whole of it, which is how `shape_census.py` counts FLOPs
and how `roof_arb` counts bytes. Summing per-op maxima over those three, times their calls, gives
the floor on exactly the byte total and exactly the FLOP census the campaign settled on. The
26-capture per-unit partition runs as a SECOND reading (`--per-unit`) because it is the one that
gives a per-unit table, and its byte total is 2.72 % higher: a standalone capture re-reads at its
own edge what the enclosing capture dedupes on buffer address. Both are reported.

Rates are `roof-shape`'s fraction of ITS cube (pc p150a, 128.70 TFLOP/s) applied to the fold's own
cube (qb2 p300c, 104.93 TFLOP/s), which is what `weigh.py` does with the aggregate. Every class
rate is the FASTEST arm measured for it, so every rate is an upper bound and the floor under it is
a lower bound.

OPS WITH NO MATMUL CLASS get an explicit rule, never the cube rate:

  eltwise, layer_norm, softmax     1 FLOP per output element, arithmetic intensity under 1
                                   FLOP/byte against a machine balance of 247.1. Traffic term
                                   only; `--eltwise-rate` prices them anyway to bound the choice.
  layout, free, bookkeeping        zero FLOPs. Traffic term only.
  matmul with no measured class    no rate exists. Bracketed by `--uncovered-lo/-hi`, defaulting
                                   to the slowest and fastest MEASURED class in this fold.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64",
       "DiffusionModule|"]


def load_modules(perf: Path):
    for d in ("roof_budget", "b2x_difflayer", "roof_residual", "roof_arb", "roof_shape"):
        sys.path.insert(0, str(perf / d))
    import exec_flops as EF
    import corrected_traffic as CT
    import split_units as SU
    import weigh as WG
    SU.CAPDIR = perf / "roof_budget" / "captures"
    return EF, CT, SU, WG


TOKEN_SDPA = "ttnn.transformer.scaled_dot_product_attention"


def token_sdpa_flops(EF, ins, outs):
    """QK^T + AV for `ttnn.transformer.scaled_dot_product_attention`.

    `exec_flops.py` has a rule for the hand-written `generic_op` SDPA and none for the stock ttnn
    one, so it falls through to "one FLOP per output element" and the fold's token attention
    arithmetic is missing from the 219.06 TFLOP census entirely. Output is (b, h, S_q, d); the
    key/value operand carries S_k. FLOPs = 4 * b * h * S_q * S_k * d.
    """
    if not outs:
        return 0.0
    o = max(outs, key=EF._prod)
    if len(o) != 4:
        return 0.0
    b, h, Sq, d = o
    kv = [s for _a, s in ins if len(s) == 4 and s[:2] == (b, h) and s[-1] == d and s != o]
    Sk = max((s[2] for s in kv), default=Sq)
    return 4.0 * b * h * Sq * Sk * d


def op_shape_rows(EF, name, ins, outs):
    """[(batch, M, K, N, flops)] for one op: `shape_census.shapes_of`, kept per op.

    Identical code path, identical tuples, so the rows join to `shape_census.json` and hence to
    `weigh.py`'s class table without a second convention.
    """
    if name in EF.FREE:
        return []
    shapes = [s for _, s in ins]
    out = []
    if name in EF.MATMUL:
        if not outs:
            return []
        sh = max(outs, key=EF._prod)
        if len(sh) < 2:
            return []
        M, N = sh[-2], sh[-1]
        batch = EF._prod(sh[:-2])
        acts = [s for s in shapes if len(s) >= 2 and EF._prod(s[:-1]) == batch * M]
        if acts:
            K = max(s[-1] for s in acts)
        else:
            w = [s for s in shapes if len(s) == 2 and s[-1] == N]
            if not w:
                return []
            K = max(s[0] for s in w)
        out.append((batch, M, K, N, 2 * batch * M * N * K))
    elif name == "ttnn.generic_op":
        w2 = [s for s in shapes if len(s) == 2]
        if w2:
            for K, N in w2:
                cand = [EF._prod(s) for s in shapes if s[-1] == K and len(s) >= 2]
                if not cand:
                    continue
                rows = max(cand) // K
                out.append((1, rows, K, N, 2 * rows * K * N))
        else:
            mask = [s for s in shapes if len(s) == 4 and s[0] == 1 and s[-1] == s[-2]]
            qkv = [s for s in shapes if len(s) == 4 and s[0] != 1 and s[-1] != s[-2]]
            if mask and qkv:
                b, h, S, d = max(qkv, key=EF._prod)
                out.append((b * h, S, d, S, 2 * b * h * S * S * d))
                out.append((b * h, S, S, d, 2 * b * h * S * S * d))
    return out


def class_rates(WG, roofs, cube_of_record):
    """census shape tuple -> (class label, FLOP/s on the part the floor of record lives on)."""
    rate = {r["arm"]: r["TFLOPs"] for r in roofs["rows"]}
    cube = roofs["cube4096_TFLOPs"]
    out = {}
    for label, shapes, arms in WG.CLASSES:
        cand = [rate[n] for n in arms if n in rate]
        if not cand:
            continue
        r = max(cand) / cube * cube_of_record
        for s in shapes:
            out[s] = (label, r)
    return out


class Join:
    """One capture, walked once: per-op bytes, FLOPs, shape rows, owner, and the two terms."""

    def __init__(self, R, sig):
        EF, CT, SU = R["EF"], R["CT"], R["SU"]
        nodes = EF.nodes_of(SU.cap_path(sig))
        self.ops, self.owner, _b, _f, _o, _r = SU.split(sig, R["have"], R["edges"])
        cnt = CT.counts({"nodes": nodes}, l1=True, pre=True, gate=True)
        self.bytes = cnt["by_op"]
        self.unattributed = cnt["unattributed_B"]
        self.real_MB = cnt["real_MB"]
        _o2, self.ins, self.outs = EF.operands(nodes)
        assert len(self.ops) == len(self.bytes) == len(self.owner)


def op_terms(R, J, i):
    """(bytes, matmul FLOPs, other FLOPs, t_traffic, t_arith_point, t_arith_lo, t_arith_hi,
        uncovered FLOPs, class label or None) for op i of capture J."""
    EF = R["EF"]
    name = J.ops[i]["name"]
    B = float(J.bytes[i])
    log, pad, kind = EF.op_flops(name, J.ins[i], J.outs[i])
    t_tr = B / R["stream"]
    tsdpa = token_sdpa_flops(EF, J.ins[i], J.outs[i]) if name == TOKEN_SDPA else 0.0
    t_ar = t_lo = t_hi = 0.0
    F_mm = unc = 0.0
    label = None
    for (b, M, K, N, f) in op_shape_rows(EF, name, J.ins[i], J.outs[i]):
        F_mm += f
        hit = R["RATES"].get((b, M, K, N))
        if hit:
            label, r = hit
            t_ar += f / r
            t_lo += f / r
            t_hi += f / r
        else:
            unc += f
            t_ar += f / R["hi_rate"]      # the point floor prices the uncovered at the fastest
            t_lo += f / R["hi_rate"]      # measured class, so it stays a floor
            t_hi += f / R["lo_rate"]
    F_el = float(pad) if kind in ("eltwise", "noshape") else 0.0
    if tsdpa and R["token_sdpa_rate"]:
        t_ar += tsdpa / R["token_sdpa_rate"]
        t_lo += tsdpa / R["hi_rate"]
        t_hi += tsdpa / R["lo_rate"]
    return B, F_mm, F_el, t_tr, t_ar, t_lo, t_hi, unc, label, tsdpa


def accumulate(R, sig, calls, agg, per_class, bucket, by_owner, unc_shapes=None,
               nonmm=None):
    J = Join(R, sig)
    for i in range(len(J.ops)):
        B, F_mm, F_el, t_tr, t_ar, t_lo, t_hi, unc, label, tsdpa = op_terms(R, J, i)
        t = max(t_tr, t_ar)
        agg["F_token_sdpa"] += calls * tsdpa
        t_el = F_el / R["eltwise_rate"] if R["eltwise_rate"] else 0.0
        agg["B"] += calls * B
        agg["F_mm"] += calls * F_mm
        agg["F_el"] += calls * F_el
        agg["unc"] += calls * unc
        agg["s_traffic"] += calls * t_tr
        agg["s_arith"] += calls * t_ar
        agg["floor"] += calls * t
        agg["floor_lo"] += calls * max(t_tr, t_lo)
        agg["floor_hi"] += calls * max(t_tr, t_hi)
        agg["floor_el"] += calls * max(t_tr, t_ar, t_el)
        agg["n_ops"] += calls
        if t_ar > t_tr:
            agg["s_by_arith"] += calls * t
            agg["n_arith"] += calls
        else:
            agg["s_by_traffic"] += calls * t
            agg["n_traffic"] += calls
        if label:
            c = per_class[label]
            c["F"] += calls * F_mm
            c["s_arith"] += calls * t_ar
            c["s_traffic"] += calls * t_tr
            c["s"] += calls * t
            c["n"] += calls
        if unc_shapes is not None and unc:
            for (b, M, K, N, f) in op_shape_rows(R["EF"], J.ops[i]["name"], J.ins[i], J.outs[i]):
                if (b, M, K, N) not in R["RATES"]:
                    e = unc_shapes[(b, M, K, N)]
                    e["n"] += calls
                    e["F"] += calls * f
        if nonmm is not None and not F_mm:
            e = nonmm[J.ops[i]["name"].replace("ttnn.", "")]
            e["n"] += calls
            e["B"] += calls * B
            e["F"] += calls * F_el
            e["s"] += calls * t
        k = "matmul" if F_mm else ("eltwise" if F_el else "layout/free")
        bk = bucket[k]
        bk["n"] += calls
        bk["F"] += calls * (F_mm or F_el)
        bk["B"] += calls * B
        bk["s_tr"] += calls * t_tr
        bk["s_ar"] += calls * t_ar
        bk["s"] += calls * t
        bk["s_el"] += calls * max(t_tr, t_ar, t_el)
        if t_ar > t_tr:
            bk["n_ar"] += calls
        o = by_owner[J.owner[i] or ("(own) " + sig.split("|")[0])]
        o["n"] += calls
        o["B"] += calls * B
        o["s_tr"] += calls * t_tr
        o["s_ar"] += calls * t_ar
        o["s"] += calls * t
    # bytes with no owning op: a pre-existing weight whose every consumer fails moves_dram. Real
    # traffic, zero FLOPs, so a pure traffic term. Never dropped from a per-op sum.
    u = J.unattributed
    agg["B"] += calls * u
    agg["s_traffic"] += calls * u / R["stream"]
    for k in ("floor", "floor_lo", "floor_hi", "floor_el", "s_by_traffic"):
        agg[k] += calls * u / R["stream"]
    bucket["layout/free"]["B"] += calls * u
    bucket["layout/free"]["s_tr"] += calls * u / R["stream"]
    bucket["layout/free"]["s"] += calls * u / R["stream"]
    by_owner["(own) " + sig.split("|")[0]]["B"] += calls * u
    by_owner["(own) " + sig.split("|")[0]]["s_tr"] += calls * u / R["stream"]
    by_owner["(own) " + sig.split("|")[0]]["s"] += calls * u / R["stream"]
    return J


def new_agg():
    return defaultdict(float)


def setup(perf, args):
    EF, CT, SU, WG = load_modules(perf)
    bud = json.loads((perf / "roof_budget" / "roof_budget_512_qb2c2.json").read_text())
    att = json.loads((perf / "roof_budget" / "attrib2_512_tip_qb2c2.json").read_text())["attrib"]
    roofs = json.loads((perf / "roof_shape" / args.roofs).read_text())
    S = bud["summary"]
    RATES = class_rates(WG, roofs, S["compute_roof_TFLOPs"] * 1e12)
    rates = [r for _l, r in RATES.values()]
    edges = set()
    for path in att["tree"]:
        p = path.split("/")
        edges.update(zip(p, p[1:]))
    return {"EF": EF, "CT": CT, "SU": SU, "WG": WG, "S": S, "roofs": roofs, "RATES": RATES,
            "by": {r["sig"]: r for r in bud["rows"]},
            "have": [s for s in att["sigs"] if SU.cap_path(s).is_file()], "edges": edges,
            "stream": S["stream_roof_GBps"] * 1e9, "cube": S["compute_roof_TFLOPs"] * 1e12,
            "balance": S["machine_balance_flop_per_byte"], "scale": S["cell_scale"],
            "lo_rate": (args.uncovered_lo * 1e12) if args.uncovered_lo else min(rates),
            "hi_rate": (args.uncovered_hi * 1e12) if args.uncovered_hi else max(rates),
            "eltwise_rate": args.eltwise_rate * 1e12 if args.eltwise_rate else 0.0,
            "token_sdpa_rate": 0.0}


def run(R, sigs, extra=False):
    agg, per_class, bucket, by_owner = (new_agg(),
                                        defaultdict(new_agg), defaultdict(new_agg),
                                        defaultdict(new_agg))
    unc_shapes = defaultdict(new_agg) if extra else None
    nonmm = defaultdict(new_agg) if extra else None
    caps = {}
    for sig, calls in sigs:
        caps[sig] = accumulate(R, sig, calls, agg, per_class, bucket, by_owner, unc_shapes, nonmm)
    if extra:
        agg["_unc_shapes"] = dict(unc_shapes)
        agg["_nonmm"] = dict(nonmm)
    return agg, dict(per_class), dict(bucket), dict(by_owner), caps



def measured_compare(R, perf, by_unit_class, per_top_floor):
    """Every class's floor beside the time that class actually takes.

    `roof-residual-census` gives each unit's OWN-work seconds at the cell; summed by class they
    tile the fold at 0.00 %. A floor above the measured time is a contradiction in the inputs, not
    in the max, so it is reported rather than hidden, and the capped total is carried as the
    conservative end of the floor.
    """
    res = json.loads((perf / "roof_residual" / "residual_512_qb2c2.json").read_text())
    meas = defaultdict(float)
    for r in res["rows"]:
        meas[r["unit"].split("|")[0]] += r["s_own_at_cell"]
    mine = defaultdict(lambda: [0.0, 0.0, 0.0])
    for k, v in by_unit_class.items():
        c = k.replace("(own) ", "")
        mine[c][0] += v["s"]
        mine[c][1] += v["s_ar"]
        mine[c][2] += v["s_tr"]
    rows = []
    for c in sorted(set(meas) | set(mine), key=lambda c: -mine[c][0]):
        f, ar, tr = mine[c]
        m = meas[c]
        rows.append({"class": c, "measured_own_at_cell_s": m, "floor_s": f,
                     "arith_s": ar, "traffic_s": tr,
                     "pct_of_measured": (100 * f / m) if m > 1e-9 else None,
                     "capped_s": min(f, m) if m > 1e-9 else f})
    return {"rows": rows, "capped_floor_s": sum(r["capped_s"] for r in rows),
            "per_top_capture": per_top_floor}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", type=Path, default=HERE.parent)
    ap.add_argument("--roofs", default="shape_roofs_pc_bh.json")
    ap.add_argument("--eltwise-rate", type=float, default=None,
                    help="TFLOP/s to price non-matmul FLOPs at, for the bound only. Default is "
                         "the slowest MEASURED matmul class in this fold, on the qb2 part.")
    ap.add_argument("--token-sdpa-rate", type=float, default=None,
                    help="TFLOP/s for ttnn.transformer.scaled_dot_product_attention, whose real "
                         "QK^T+AV arithmetic exec_flops.py does not count. Off by default so the "
                         "headline floor stays exactly on the settled 219.058 TFLOP census; the "
                         "correction is reported beside it either way.")
    ap.add_argument("--uncovered-lo", type=float, default=None)
    ap.add_argument("--uncovered-hi", type=float, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "true_floor_512_qb2c2.json")
    a = ap.parse_args()

    perf = a.perf.resolve()
    R = setup(perf, a)
    if not a.eltwise_rate:
        R["eltwise_rate"] = R["lo_rate"]
    R["token_sdpa_rate"] = (a.token_sdpa_rate * 1e12) if a.token_sdpa_rate else 0.0
    sdpa_class = R["RATES"][(2048, 512, 32, 512)][1]
    S, by, scale, cell = R["S"], R["by"], R["scale"], R["S"]["cell_of_record_s"]

    A, per_class, bucket, by_owner, caps = run(R, [(t, by[t]["calls"]) for t in TOP], extra=True)
    U, _pc, _bk, _bo, ucaps = run(R, [(s, by[s]["calls"]) for s in sorted(by)])
    # the second reading is the unit's OWN work only, so re-walk it with the child ops dropped
    Uo = new_agg()
    for sig in sorted(by):
        J = ucaps[sig]
        calls = by[sig]["calls"]
        for i in range(len(J.ops)):
            if J.owner[i] != "":
                continue
            B, F_mm, F_el, t_tr, t_ar, t_lo, t_hi, unc, _l, _t = op_terms(R, J, i)
            Uo["B"] += calls * B
            Uo["floor"] += calls * max(t_tr, t_ar)
            Uo["s_traffic"] += calls * t_tr
            Uo["s_arith"] += calls * t_ar
        Uo["B"] += calls * J.unattributed
        Uo["floor"] += calls * J.unattributed / R["stream"]
        Uo["s_traffic"] += calls * J.unattributed / R["stream"]

    # the one class with real FLOPs and no measured rate: bounded, never assumed
    tsd = []
    for nm, r in (("fastest measured class", R["hi_rate"]),
                  ("fused triangle SDPA class", sdpa_class),
                  ("slowest measured class", R["lo_rate"])):
        R["token_sdpa_rate"] = r
        At, _a, _b, _c, _d = run(R, [(t, by[t]["calls"]) for t in TOP])
        tsd.append({"rate_name": nm, "TFLOPs": r / 1e12, "floor_s": At["floor"],
                    "TFLOP_per_fold": At["F_token_sdpa"] / 1e12})
    R["token_sdpa_rate"] = (a.token_sdpa_rate * 1e12) if a.token_sdpa_rate else 0.0

    per_top = []
    for t in TOP:
        At, _a, _b, _c, _d = run(R, [(t, by[t]["calls"])])
        m = by[t]["s_per_fold"] * scale
        per_top.append({"capture": t, "calls": by[t]["calls"], "measured_at_cell_s": m,
                        "floor_s": At["floor"], "traffic_s": At["s_traffic"],
                        "arith_s": At["s_arith"], "pct_of_measured": 100 * At["floor"] / m})

    top_fold_s = sum(by[t]["s_per_fold"] for t in TOP)
    top_cell = top_fold_s * scale
    census = json.loads((perf / "roof_budget" / "shape_census.json").read_text())
    out_by_owner = dict(sorted(by_owner.items(), key=lambda kv: -kv[1]["s"]))

    out = {
        "granularity": ("per top-level ttnn op, in the three disjoint top-level captures, "
                        "summed over their calls per fold"),
        "roofs": {"stream_GBps": S["stream_roof_GBps"],
                  "dense_cube_TFLOPs": S["compute_roof_TFLOPs"],
                  "shape_rate_host": R["roofs"]["host"],
                  "shape_rate_cube_TFLOPs": R["roofs"]["cube4096_TFLOPs"],
                  "machine_balance_flop_per_byte": S["machine_balance_flop_per_byte"],
                  "cell_of_record_s": cell, "cell_scale": scale,
                  "uncovered_bracket_TFLOPs": [R["lo_rate"] / 1e12, R["hi_rate"] / 1e12],
                  "eltwise_bound_rate_TFLOPs": R["eltwise_rate"] / 1e12},
        "floor_s": A["floor"], "floor_lo_s": A["floor_lo"], "floor_hi_s": A["floor_hi"],
        "floor_if_eltwise_flops_priced_s": A["floor_el"],
        "traffic_only_s": A["s_traffic"], "arithmetic_only_s": A["s_arith"],
        "s_set_by_arithmetic": A["s_by_arith"], "s_set_by_traffic": A["s_by_traffic"],
        "ops_per_fold": A["n_ops"], "ops_arithmetic_bound": A["n_arith"],
        "ops_traffic_bound": A["n_traffic"],
        "prize_s": cell - A["floor"],
        "prize_of_record_s": cell - 11.134,
        "fold_TB": A["B"] / 1e12, "matmul_TFLOP": A["F_mm"] / 1e12,
        "nonmatmul_TFLOP": A["F_el"] / 1e12,
        "uncovered_matmul_TFLOP": A["unc"] / 1e12,
        "uncovered_pct_of_matmul_FLOP": 100 * A["unc"] / A["F_mm"],
        "nonmatmul_floor_s": bucket["eltwise"]["s"] + bucket["layout/free"]["s"],
        "matmul_floor_s": bucket["matmul"]["s"],
        "calibration": {
            "bytes_vs_roof_arb_TB": [A["B"] / 1e12, 2.8589],
            "matmul_FLOP_vs_shape_census_TFLOP": [A["F_mm"] / 1e12, census["total_TFLOP"]],
            "top_level_s_per_fold": top_fold_s, "top_level_at_cell_s": top_cell,
            "cell_of_record_s": cell,
            "per_unit_second_reading": {"fold_TB": Uo["B"] / 1e12, "floor_s": Uo["floor"],
                                        "traffic_only_s": Uo["s_traffic"],
                                        "arithmetic_only_s": Uo["s_arith"]},
            "all_ops_26_captures_TB": U["B"] / 1e12},
        "buckets": bucket, "per_class": per_class,
        "uncovered_shapes": {"%d,%d,%d,%d" % k: v for k, v in
                             sorted(A["_unc_shapes"].items(), key=lambda kv: -kv[1]["F"])},
        "non_matmul_ops": dict(sorted(A["_nonmm"].items(), key=lambda kv: -kv[1]["s"])),
        "token_sdpa_correction": tsd,
        "vs_measured": measured_compare(R, perf, out_by_owner, per_top),
        "by_unit_class": out_by_owner,
    }
    a.out.write_text(json.dumps(out, indent=1, default=float))

    p = print
    p("granularity: %s" % out["granularity"])
    p("roofs: stream %.1f GB/s, dense cube %.2f TFLOP/s (qb2 p300c). Shape-honest fractions from "
      "%s, cube %.2f TFLOP/s." % (S["stream_roof_GBps"], S["compute_roof_TFLOPs"],
                                  R["roofs"]["host"], R["roofs"]["cube4096_TFLOPs"]))
    p("")
    p("%-34s %9s %11s %10s %10s %10s" % ("unit class (deepest open frame)", "ops/fold", "TB",
                                         "s traffic", "s arith", "s MAX"))
    for k, v in out["by_unit_class"].items():
        p("%-34s %9d %11.4f %10.4f %10.4f %10.4f"
          % (k, v["n"], v["B"] / 1e12, v["s_tr"], v["s_ar"], v["s"]))
    p("")
    p("%-14s %9s %11s %11s %10s %10s %10s %9s" % ("kind", "ops/fold", "TB", "TFLOP", "s traffic",
                                                  "s arith", "s MAX", "arith-bnd"))
    for k, v in sorted(bucket.items(), key=lambda kv: -kv[1]["s"]):
        p("%-14s %9d %11.4f %11.2f %10.4f %10.4f %10.4f %9d"
          % (k, v["n"], v["B"] / 1e12, v["F"] / 1e12, v["s_tr"], v["s_ar"], v["s"], v["n_ar"]))
    p("")
    p("%-58s %9s %9s %9s" % ("matmul class", "TFLOP", "s arith", "s MAX"))
    for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]["s"]):
        p("%-58s %9.2f %9.4f %9.4f" % (k, v["F"] / 1e12, v["s_arith"], v["s"]))
    p("")
    p("traffic-only aggregate        %8.3f s   (%.4f TB / %.1f GB/s)"
      % (A["s_traffic"], A["B"] / 1e12, S["stream_roof_GBps"]))
    cov_s = sum(v["s_arith"] for v in per_class.values())
    cov_F = sum(v["F"] for v in per_class.values())
    eff = cov_F / cov_s
    p("arithmetic-only aggregate     %8.3f s   (%.2f matmul TFLOP at shape-honest class rates)"
      % (A["s_arith"], A["F_mm"] / 1e12))
    p("   covered %.2f TFLOP in %.3f s = %.2f TFLOP/s = %.2f %% of the %.2f TFLOP/s cube, which "
      "is roof-shape's 18.8 %%" % (cov_F / 1e12, cov_s, eff / 1e12, 100 * eff / R["cube"],
                                   S["compute_roof_TFLOPs"]))
    p("   the uncovered %.2f TFLOP at the fastest measured class %.2f TFLOP/s adds %.3f s; at the "
      "covered effective rate it would add %.3f s, which is the 11.134 s of record"
      % (A["unc"] / 1e12, R["hi_rate"] / 1e12, A["unc"] / R["hi_rate"], A["unc"] / eff))
    p("TRUE FLOOR, max per op        %8.3f s   bracket [%.3f, %.3f] from the uncovered %.2f %% "
      "of matmul FLOPs" % (A["floor"], A["floor_lo"], A["floor_hi"],
                           out["uncovered_pct_of_matmul_FLOP"]))
    p("  set by arithmetic           %8.3f s  (%.1f %%, %d of %d op calls)"
      % (A["s_by_arith"], 100 * A["s_by_arith"] / A["floor"], A["n_arith"], A["n_ops"]))
    p("  set by traffic              %8.3f s  (%.1f %%, %d of %d op calls)"
      % (A["s_by_traffic"], 100 * A["s_by_traffic"] / A["floor"], A["n_traffic"], A["n_ops"]))
    p("  non-matmul, in neither published floor as a term: %.3f s"
      % out["nonmatmul_floor_s"])
    p("  if non-matmul FLOPs are priced at %.2f TFLOP/s too: %.3f s (+%.3f)"
      % (R["eltwise_rate"] / 1e12, A["floor_el"], A["floor_el"] - A["floor"]))
    p("")
    p("PRIZE  %.3f s cell - %.3f s floor = %.3f s   (replaces %.3f s against the 11.134 s "
      "arithmetic aggregate)" % (cell, A["floor"], cell - A["floor"], cell - 11.134))
    p("")
    p("THE ONE CLASS WITH REAL FLOPs AND NO MEASURED RATE: ttnn.transformer.scaled_dot_product_"
      "attention")
    p("  exec_flops.py charges it 1 FLOP per output element, so %.3f TFLOP/fold of QK^T+AV is "
      "missing from the 219.058 TFLOP census and from the floor of record." % tsd[0]["TFLOP_per_fold"])
    for r in tsd:
        p("  priced at the %-26s %6.2f TFLOP/s -> floor %.3f s (%+.3f)"
          % (r["rate_name"], r["TFLOPs"], r["floor_s"], r["floor_s"] - A["floor"]))
    p("")
    p("OPS WITH NO MATMUL CLASS -- the rule is the traffic term, and this is what it covers")
    p("%-34s %11s %11s %11s %10s" % ("ttnn op", "calls/fold", "TB", "TFLOP", "s"))
    for k, v in list(out["non_matmul_ops"].items())[:14]:
        p("%-34s %11d %11.4f %11.4f %10.4f" % (k, v["n"], v["B"] / 1e12, v["F"] / 1e12, v["s"]))
    p("%-34s %11d %11.4f %11.4f %10.4f"
      % ("ALL non-matmul", sum(v["n"] for v in out["non_matmul_ops"].values()),
         sum(v["B"] for v in out["non_matmul_ops"].values()) / 1e12,
         sum(v["F"] for v in out["non_matmul_ops"].values()) / 1e12,
         sum(v["s"] for v in out["non_matmul_ops"].values())))
    p("")
    p("MATMUL SHAPES WITH NO MEASURED CLASS -- bracketed, never priced at the cube")
    p("%-34s %11s %11s" % ("batch,M,K,N", "calls/fold", "TFLOP"))
    for k, v in list(out["uncovered_shapes"].items())[:10]:
        p("%-34s %11d %11.4f" % (k, v["n"], v["F"] / 1e12))
    p("%-34s %11d %11.4f  -> %.3f s at %.2f TFLOP/s, %.3f s at %.2f TFLOP/s"
      % ("ALL uncovered (%d shapes)" % len(out["uncovered_shapes"]),
         sum(v["n"] for v in out["uncovered_shapes"].values()),
         sum(v["F"] for v in out["uncovered_shapes"].values()) / 1e12,
         A["unc"] / R["hi_rate"], R["hi_rate"] / 1e12, A["unc"] / R["lo_rate"],
         R["lo_rate"] / 1e12))

    V = out["vs_measured"]
    p("")
    p("THE FLOOR BESIDE THE TIME EACH CLASS ACTUALLY TAKES (own work at the cell, roof-residual)")
    p("%-30s %11s %11s %11s %9s" % ("class", "measured s", "floor s", "of which ar", "% meas"))
    for r in V["rows"]:
        p("%-30s %11.3f %11.3f %11.3f %9s"
          % (r["class"], r["measured_own_at_cell_s"], r["floor_s"], r["arith_s"],
             ("%.0f %%" % r["pct_of_measured"]) if r["pct_of_measured"] else "-"))
    p("%-30s %11.3f %11.3f" % ("TOTAL", sum(r["measured_own_at_cell_s"] for r in V["rows"]),
                               A["floor"]))
    p("")
    for r in V["per_top_capture"]:
        p("  %-52s x%-4d measured %7.3f s  floor %7.3f s  %5.1f %%"
          % (r["capture"], r["calls"], r["measured_at_cell_s"], r["floor_s"],
             r["pct_of_measured"]))
    p("  capping every class at its own measured time gives the conservative floor %.3f s, "
      "prize %.3f s" % (V["capped_floor_s"], cell - V["capped_floor_s"]))
    p("")
    p("CALIBRATION")
    p("  bytes        %.4f TB against roof-byte-arbitration's 2.8589 TB   %+.3f %%"
      % (A["B"] / 1e12, 100 * (A["B"] / 1e12 - 2.8589) / 2.8589))
    p("  matmul FLOPs %.3f TFLOP against shape_census's %.3f TFLOP        %+.3f %%"
      % (A["F_mm"] / 1e12, census["total_TFLOP"],
         100 * (A["F_mm"] / 1e12 - census["total_TFLOP"]) / census["total_TFLOP"]))
    p("  op set       the 3 top-level captures at %d/%d/%d calls = %.3f s of session fold, "
      "%.3f s at the cell, against the %.3f s cell of record  %+.2f %%"
      % (by[TOP[0]]["calls"], by[TOP[1]]["calls"], by[TOP[2]]["calls"], top_fold_s, top_cell,
         cell, 100 * (top_cell - cell) / cell))
    p("  2nd reading  the 26-capture per-unit own-work partition: %.4f TB (%+.2f %%), "
      "floor %.3f s (%+.2f %%)"
      % (Uo["B"] / 1e12, 100 * (Uo["B"] - A["B"]) / A["B"], Uo["floor"],
         100 * (Uo["floor"] - A["floor"]) / A["floor"]))
    p("  above both aggregates: %.3f s > 11.134 s arithmetic, > 6.732 s traffic  -- %s"
      % (A["floor"], "yes" if A["floor"] > 11.134 else "NO, THE JOIN IS WRONG"))
    p("\nWROTE %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
