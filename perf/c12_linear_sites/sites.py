#!/usr/bin/env python3
"""The 27 `linear` census keys, mapped to the module that issues them, and the sibling sets in them.

No device and no new measurement. The committed instruments are reused verbatim:
`roof_true/true_floor.py` for the capture walk, the launch arm and the census key string (so a key
here is the same string `c10-fold-census` priced), `roof_residual/split_units.py` for which marked
`unit::<Class>` frame owns each op, `roof_budget/exec_flops.py` for operand (address, shape) pairs.
The join is the one both sides already carry: capture + op index, weighted by the fold's own call
count for the three disjoint top-level captures (264 PairformerLayer + 16 MSALayer + 200
DiffusionModule). The join is checked, not asserted: every key's reconstructed call count must equal
the census's to better than half a call, and all 27 keys must be reproduced.

A SIBLING SET is >= 2 ttnn.linear calls in one owner frame that read the SAME activation buffer
ADDRESS at the same (batch, M, K) with DISTINCT weight buffers. Those are the calls one wider matmul
could issue: identical arithmetic, the shared activation read once instead of n times.

Address reuse is real (ttnn recycles a DRAM address after a deallocate, and a chunked loop reuses
one address every iteration), so a set is bounded two ways, both exact rather than heuristic:

  * one FRAME INSTANCE. `split_units` reports every `unit::` region it closed, by op-index range,
    so two calls in different instances of the same class can never join even when ttnn handed
    them the same address.
  * NO WEIGHT ADDRESS TWICE. A chunked loop inside one frame (the pair transition runs 32 row
    chunks inside a single Transition frame) repeats its weights every iteration, so a repeat ends
    the set. Without this rule the pair transition reads as one 64-member set, which it is not.

No op-index window is used. A window has to be large enough to span an attention body, which is
also large enough to merge two instances, and at 40 it silently merged 24 DiT layers into one set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from math import prod
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
sys.path.insert(0, str(PERF / "roof_true"))
import true_floor as TF                                                       # noqa: E402
sys.path.insert(0, str(PERF / "b2x_difflayer"))
from itemize import top_level_spans                                           # noqa: E402

ALLOCS = {}
DRAM_ROOF_GBs = 442.9        # c10-fold-census, same chip, same session, starved 8192^2 add
CLOCK_MHZ = 1350.0


class Args:
    roofs = "shape_roofs_qb2c3_shipped.json"
    eltwise_rate = None
    uncovered_lo = None
    uncovered_hi = None


def operand_meta(nodes):
    """op index -> {address: (shape, size_bytes, dtype)} from the capture's own tensor nodes."""
    ops, owner = top_level_spans(nodes)
    meta = defaultdict(dict)
    for n in nodes:
        p = n.get("params") or {}
        if n["node_type"] != "tensor" or not p.get("shape"):
            continue
        sh = tuple(int(x) for x in re.findall(r"\d+", str(p["shape"])))
        for c in (n.get("connections") or []):
            o = owner.get(c)
            if o is not None:
                meta[o][p.get("address")] = (sh, int(p.get("size", 0) or 0),
                                             str(p.get("dtype", "")),
                                             str(p.get("buffer_type", "")),
                                             p.get("tensor_id"))
    return meta


def split_operands(ins, out):
    """The (activation, weight, bias) split of one ttnn.linear, or None if it does not resolve."""
    if not out or len(out) < 2:
        return None
    M, N = out[-2], out[-1]
    batch = prod(out[:-2]) if len(out) > 2 else 1
    w = [(a, s) for a, s in ins if len(s) == 2 and s[-1] == N]
    if not w:
        return None
    wa, ws = max(w, key=lambda t: t[1][0])
    K = ws[0]
    act = [(a, s) for a, s in ins if len(s) >= 2 and prod(s[:-1]) == batch * M and s[-1] == K]
    if not act:
        return None
    aa, ash = max(act, key=lambda t: prod(t[1]))
    return {"act_addr": aa, "act_shape": list(ash), "w_addr": wa, "w_shape": list(ws),
            "bias": any(len(s) == 1 and s[0] == N for _a, s in ins),
            "batch": batch, "M": M, "K": K, "N": N}


def frame_instances(R, sig, ops_n):
    """op index -> id of the innermost `unit::` frame instance it ran in, from split_units'
    own region report. '' for an op in the captured unit itself and in no marked child."""
    _o, _ow, _b, _f, _orph, report = R["SU"].split(sig, R["have"], R["edges"])
    inst = [("", 0)] * ops_n
    for j, (u, start, length, _how) in enumerate(report):
        for i in range(start, min(start + length, ops_n)):
            if inst[i][1] <= start:
                inst[i] = ((u, j), start)
    return [x[0] for x in inst]


def alloc_index(nodes):
    """address -> sorted op indices at which that address was (re)allocated.

    An address is a valid identity for a tensor only until something else is allocated there.
    Two reads of one address either side of an allocation are two different tensors, which is why
    a cross-frame set has to check this and not just compare addresses."""
    ops, owner = top_level_spans(nodes)
    out = defaultdict(list)
    for n in nodes:
        if n["node_type"] != "buffer_allocate":
            continue
        p = n.get("params") or {}
        o = owner.get(n["counter"])
        if p.get("address") is not None and o is not None:
            out[p["address"]].append(o)
    return {a: sorted(v) for a, v in out.items()}


def collect(R):
    EF, SU = R["EF"], R["SU"]
    rows, addr_shapes = [], defaultdict(set)
    for sig in TF.TOP:
        calls = R["by"][sig]["calls"]
        J = TF.Join(R, sig)
        meta = operand_meta(EF.nodes_of(SU.cap_path(sig)))
        inst = frame_instances(R, sig, len(J.ops))
        allocs = alloc_index(EF.nodes_of(SU.cap_path(sig)))
        for i, op in enumerate(J.ops):
            if op["name"] != "ttnn.linear":
                continue
            ins, outs = J.ins[i], J.outs[i]
            arm = TF.launch_arm(op["name"], ins)
            mmrows = TF.op_shape_rows(EF, op["name"], ins, outs)
            if TF.launch_key(arm, op["name"], ins, outs, mmrows) is None:
                continue
            sh = TF.launch_shape(op["name"], ins, outs)
            K = mmrows[0][2] if mmrows else None
            row = {"sig": sig, "i": i, "calls": calls, "owner": J.owner[i],
                   "inst": "%s#%d" % inst[i] if inst[i] else "",
                   "key": "%s|out=%s|K=%s" % (arm, "x".join(str(d) for d in sh), K),
                   "B": float(J.bytes[i])}
            o = split_operands(ins, list(sh))
            if o is None:
                row["unresolved"] = True
            else:
                row.update(o)
                am = meta.get(i, {}).get(o["act_addr"])
                row["act_B"] = am[1] if am else 0
                row["act_dtype"] = am[2].replace("DataType::", "") if am else "?"
                row["act_mem"] = am[3].replace("BufferType::", "") if am else "?"
                wm = meta.get(i, {}).get(o["w_addr"])
                row["w_B"] = wm[1] if wm else 0
                # the weight's TENSOR id, not its address: PairWeightedAveraging slices 16
                # per-head weights out of two parent tensors one at a time, and ttnn hands every
                # slice the SAME recycled address. Keyed on address, the distinctness rule below
                # split all 16 apart and dropped a real 256-call set. Bytes still dedupe on
                # address; only "is this a different weight" reads the tensor id.
                row["w_tid"] = wm[4] if wm else None
                am2 = meta.get(i, {}).get(o["act_addr"])
                row["act_tid"] = am2[4] if am2 else None
                for a, s in ins:
                    addr_shapes[a].add(tuple(s))
            rows.append(row)
        ALLOCS[sig] = allocs
    return rows, {a: sorted(map(list, v)) for a, v in addr_shapes.items() if len(v) > 1}


def sibling_sets(rows, window):
    """Frame-local sets. Two guards, both needed:

      * no reallocation of the activation's address between members. A loop that frees and
        refills one buffer (the pair transition's 32 row chunks, PairWeightedAveraging's 8
        per-head attention outputs) hands every iteration the same address, and without this the
        iterations read as one set.
      * no repeat of the same weight TENSOR.
    """
    by = defaultdict(list)
    for r in rows:
        if r.get("unresolved"):
            continue
        by[(r["sig"], r["inst"], r["act_addr"], tuple(r["act_shape"]), r["K"])].append(r)
    sets = []
    for k, group in by.items():
        group.sort(key=lambda r: r["i"])
        al = ALLOCS.get(k[0], {}).get(group[0]["act_addr"], [])
        run = [group[0]]
        for r in group[1:]:
            fresh = any(run[-1]["i"] < x <= r["i"] for x in al)
            if not fresh and (r["w_addr"], r["w_tid"]) not in {(q["w_addr"], q["w_tid"])
                                                               for q in run}:
                run.append(r)
            else:
                if len(run) > 1:
                    sets.append((k, run))
                run = [r]
        if len(run) > 1:
            sets.append((k, run))
    return sets


def cross_frame_sets(rows):
    """Sets that span frame instances: n linears reading ONE tensor that nothing reallocated.

    The DiT's `s` is the case this exists for: it is computed once per sampling step and read by
    all 24 layers' output projections, so those 48 calls are siblings even though each sits in its
    own frame. Validity test, not an assumption: no `buffer_allocate` of that address may fall
    between the first and last member, and the weight addresses must all differ."""
    by = defaultdict(list)
    for r in rows:
        if r.get("unresolved"):
            continue
        by[(r["sig"], r["act_addr"], tuple(r["act_shape"]), r["K"])].append(r)
    out = []
    for (sig, addr, ash, K), group in by.items():
        group.sort(key=lambda r: r["i"])
        if len({r["inst"] for r in group}) < 2:
            continue
        al = ALLOCS.get(sig, {}).get(addr, [])
        run = [group[0]]
        for r in group[1:]:
            reallocated = any(run[-1]["i"] < x <= r["i"] for x in al)
            if reallocated or (r["w_addr"], r["w_tid"]) in {(q["w_addr"], q["w_tid"])
                                                            for q in run}:
                if len({q["inst"] for q in run}) > 1:
                    out.append(((sig, addr, ash, K), run))
                run = [r]
            else:
                run.append(r)
        if len({q["inst"] for q in run}) > 1:
            out.append(((sig, addr, ash, K), run))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--out", type=Path, default=HERE / "sites_512.json")
    a = ap.parse_args()

    R = TF.setup(PERF, Args())
    rows, reuse = collect(R)
    budget = json.loads((HERE / "census_budget_sweep2.json").read_text())
    cen = {k["key"]: k for k in budget["keys"] if k["arm"] == "linear"}

    per_key = defaultdict(lambda: {"calls": 0.0, "sites": defaultdict(float)})
    for r in rows:
        e = per_key[r["key"]]
        e["calls"] += r["calls"]
        tag = "%s/%s" % (r["sig"].split("|")[0], r["owner"] or "(self)")
        e["sites"][tag] += r["calls"]

    print("== CALL-SITES: census key -> owning unit ==")
    bad = []
    tot_c = tot_s = 0.0
    for key, e in sorted(per_key.items(), key=lambda kv: -cen.get(kv[0], {}).get("fold_s", 0)):
        c = cen.get(key)
        if not c or abs(c["calls"] - e["calls"]) >= 0.5:
            bad.append(key)
        print("%-40s %8.0f calls  %7.4f s  %8.1f Mc  %6.1f GB/s  %5.1f TFLOP/s" %
              (key, e["calls"], c["fold_s"], c["Mcycles"], c["GBs"], c["TFLOPs"]))
        for tag, n in sorted(e["sites"].items(), key=lambda kv: -kv[1]):
            print("      %-52s %8.0f (%4.1f%%)" % (tag, n, 100 * n / e["calls"]))
        tot_c += c["calls"]
        tot_s += c["fold_s"]
    print("JOIN: %d keys, %.0f calls, %.4f s" % (len(per_key), tot_c, tot_s))
    miss = [k for k in cen if k not in per_key]
    if bad or miss or len(per_key) != 27:
        print("JOIN BROKEN: mismatched %s missing %s" % (bad, miss))
        return 1

    sets = sibling_sets(rows, a.window)
    agg = defaultdict(lambda: {"n": 0, "members": None, "linear_calls": 0.0, "s": 0.0,
                               "reread_B": 0.0, "total_B": 0.0, "flop": 0.0, "act_B": 0,
                               "act_mem": "?", "act_dtype": "?", "gap": 0})
    for (sig, _inst, _addr, ash, K), run in sets:
        owner = run[0]["owner"]
        members = tuple(sorted((r["N"], r["key"], r["w_B"]) for r in run))
        gk = (sig.split("|")[0], owner, tuple(ash), K, members)
        e = agg[gk]
        e["members"] = members
        e["n"] += 1
        calls = run[0]["calls"]
        e["act_B"] = run[0]["act_B"]
        e["act_mem"] = run[0].get("act_mem", "?")
        e["act_dtype"] = run[0].get("act_dtype", "?")
        e["gap"] = max(e.get("gap", 0), run[-1]["i"] - run[0]["i"])
        e["linear_calls"] += calls * len(run)
        e["reread_B"] += calls * (len(run) - 1) * run[0]["act_B"]
        for r in run:
            c = cen[r["key"]]
            e["s"] += calls * c["us_per_call"] * 1e-6
            # bytes from the CENSUS's own per-key figure (B = calls*(in+out tiles)*2048), scaled to
            # this member's call share. The capture's per-op byte column dedupes a re-read buffer
            # across ops inside one capture, so summing it would hide the very re-read being priced.
            e["total_B"] += calls * c["GB"] * 1e9 / c["calls"]
            e["flop"] += calls * c["TFLOP"] * 1e12 / c["calls"]
    out = []
    print("\n== SIBLING-SETS ==")
    print("%-16s %-26s %-20s %4s %8s %8s %9s %9s %8s %7s %6s" %
          ("capture", "owner", "act", "n", "calls", "fold_s", "Mcycles", "reread_GB",
           "total_GB", "GB/s", "act_mem"))
    for gk, e in sorted(agg.items(), key=lambda kv: -kv[1]["s"]):
        cap, owner, ash, K, members = gk
        gbs = e["total_B"] / e["s"] / 1e9
        mc = e["s"] * CLOCK_MHZ
        # byte-proportional prediction at the measured achieved rate: the conservative floor
        b_new = e["total_B"] - e["reread_B"]
        s_bytes = b_new / (gbs * 1e9)
        out.append({"capture": cap, "owner": owner, "act_shape": list(ash), "K": K,
                    "members": [{"N": n, "key": k, "weight_B": w} for n, k, w in members],
                    "instances_per_capture": e["n"], "linear_calls": e["linear_calls"],
                    "fold_s": e["s"], "Mcycles": mc, "reread_B": e["reread_B"],
                    "total_B": e["total_B"], "GBs": gbs, "TFLOPs": e["flop"] / e["s"] / 1e12,
                    "act_B": e["act_B"], "act_mem": e["act_mem"],
                    "act_dtype": e["act_dtype"], "op_gap": e["gap"],
                    "s_at_constant_rate": s_bytes,
                    "win_at_constant_rate_s": e["s"] - s_bytes})
        print("%-16s %-26s %-20s %4d %8.0f %8.4f %9.1f %9.2f %8.2f %7.1f %6s" %
              (cap, owner or "(self)", "x".join(map(str, ash)) + " K%d" % K, e["n"],
               e["linear_calls"], e["s"], mc, e["reread_B"] / 1e9, e["total_B"] / 1e9, gbs,
               e["act_mem"]))
        shown = members if len(members) <= 4 else members[:2]
        for n, k, w in shown:
            print("      N=%-6s %-40s w %5.2f MB" % (n, k, w / 1e6))
        if len(members) > 4:
            print("      ... %d more members, same shape" % (len(members) - 2))
    xf = cross_frame_sets(rows)
    xout = []
    print("\n== CROSS-FRAME SETS (one tensor, no reallocation between members) ==")
    for (sig, addr, ash, K), run in sorted(xf, key=lambda t: -len(t[1])):
        c0 = run[0]["calls"]
        secs = sum(c0 * cen[r["key"]]["us_per_call"] * 1e-6 for r in run)
        reread = c0 * (len(run) - 1) * run[0]["act_B"]
        owners = sorted({r["owner"] or "(self)" for r in run})
        print("%-16s %-44s n=%-3d calls %7.0f  fold_s %7.4f  Mc %8.1f  reread %5.2f GB  %s"
              % (sig.split("|")[0], "x".join(map(str, ash)) + " K%d" % K + " N_fused=%d"
                 % sum(r["N"] for r in run), len(run), c0 * len(run), secs, secs * CLOCK_MHZ,
                 reread / 1e9, ",".join(owners)))
        xout.append({"sig": sig, "act_shape": list(ash), "K": K, "n": len(run),
                     "N_fused": sum(r["N"] for r in run), "calls": c0 * len(run),
                     "fold_s": secs, "reread_B": reread, "owners": owners,
                     "act_mem": run[0].get("act_mem"),
                     "members": [{"N": r["N"], "key": r["key"], "owner": r["owner"],
                                  "i": r["i"]} for r in run]})
    json.dump({"clock_MHz": CLOCK_MHZ, "census": "c10-fold-census sweep2 budget.json",
               "keys": {k: {"calls": v["calls"], "sites": dict(v["sites"])}
                        for k, v in per_key.items()},
               "sets": out, "cross_frame_sets": xout, "address_reuse": reuse, "rows": rows},
              open(a.out, "w"), indent=1, default=float)
    print("\nlinear ops walked: %d; addresses seen with >1 shape: %d" % (len(rows), len(reuse)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
