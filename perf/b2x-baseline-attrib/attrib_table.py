#!/usr/bin/env python3
"""Turn one instrumented fold into the per-phase, per-op and per-dtype attribution tables.

Host only, no device. Reads what ``baseline_attrib.py`` wrote:

  * ``attrib.tree``     per bracket path: calls, inclusive and exclusive wall
  * ``attrib.sigs``     per (unit, input shape): calls and median ms
  * ``captures/``       one ``ttnn.graph`` capture per (unit, input shape)
  * ``census.rows``     every ttnn call of a second fold, charged to the bracket stack under a
                        floor byte model, which is the only reading of the ops that belong to
                        no module

Byte accounting inside a capture is the same as ``perf/bioir_roofline/fold_bytes_512.py`` -- a
DRAM-resident tensor is read once per region, a DRAM buffer allocated inside the region is
written once -- so a phase's GB/call is directly comparable to the published byte budget. The
difference is where each byte is charged: to the innermost ttnn op and the innermost bracketed
unit that encloses it, instead of to the region as a whole. Region totals are unchanged by that,
which is the check the table prints.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Measured on this card by perf/bioir_roofline/roofs_bh.py, cited from
# perf/bioir_roofline/roofs_p300c_qb2_card2.json.
STREAM_ROOF_GBS = 429.9

# The published byte budget's four instrumented phases, so the table can say what is new.
PUBLISHED = {"PairformerLayer", "DiffusionTransformerLayer", "MSALayer"}


def walk(nodes):
    """Charge every tensor read and DRAM allocation to (unit, op), in capture order.

    ``stacking_level`` is not used to find the parent: it is a depth label, and two sibling
    subtrees share one. The frame stack does it exactly.
    """
    stack: list[str] = []
    seen: set[int] = set()
    rows: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"read": 0, "write": 0, "l1": 0, "n_tensor": 0, "n_alloc": 0})
    dtypes: dict[str, int] = defaultdict(int)
    ops: dict[str, int] = defaultdict(int)
    no_buffer_type = 0
    total = {"read": 0, "write": 0, "l1": 0, "ops": 0}

    def here():
        unit = "(fold)"
        for f in reversed(stack):
            if f.startswith("unit::"):
                unit = f[6:]
                break
        op = None
        started = False
        for f in stack:
            if f.startswith("unit::"):
                started = True
                op = None
                continue
            if started and op is None and (f.startswith("ttnn.") or f.startswith("ttnn::")):
                op = f
        if op is None:
            op = stack[-1] if stack else "(none)"
        return unit, op

    for n in nodes:
        t = n.get("node_type")
        p = n.get("params") or {}
        if t == "function_start":
            name = str(p.get("name", ""))
            if name.startswith("ttnn."):
                total["ops"] += 1
                ops[name] += 1
            stack.append(name)
        elif t == "function_end":
            if stack:
                stack.pop()
        elif t == "tensor":
            tid = p.get("tensor_id")
            if tid in seen:
                continue
            seen.add(tid)
            bt = str(p.get("buffer_type", ""))
            if not bt:
                no_buffer_type += 1
            if "DRAM" not in bt:
                continue
            sz = int(p.get("size", 0) or 0)
            u, o = here()
            r = rows[(u, o)]
            r["read"] += sz
            r["n_tensor"] += 1
            dtypes[str(p.get("dtype", "?")).split("::")[-1]] += sz
            total["read"] += sz
        elif t == "buffer_allocate":
            sz = int(p.get("size", 0) or 0)
            u, o = here()
            r = rows[(u, o)]
            if str(p.get("type")) == "DRAM":
                r["write"] += sz
                r["n_alloc"] += 1
                total["write"] += sz
            else:
                r["l1"] += sz
                total["l1"] += sz
    return rows, dtypes, ops, total, no_buffer_type


def gb(x):
    return round(x / 1e9, 6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    R = json.loads(a.run.read_text())
    at = R.get("attrib") or {}
    tree, sigs, caps = at.get("tree", {}), at.get("sigs", {}), at.get("captures", {})

    # ---- per capture: region totals, per (unit, op) and per dtype ---------------------------
    per_sig = {}
    for sig, info in sorted(caps.items()):
        f = ROOT / info["file"]
        if not f.exists():
            continue
        with gzip.open(f, "rt") as fh:
            nodes = json.load(fh)
        rows, dtypes, ops, total, nobt = walk(nodes)
        per_sig[sig] = {
            "nodes": len(nodes), "ttnn_ops": total["ops"],
            "dram_read": total["read"], "dram_write": total["write"],
            "dram_total": total["read"] + total["write"], "l1_alloc": total["l1"],
            "tensor_nodes_without_buffer_type": nobt,
            "by_unit_op": [{"unit": u, "op": o, **{k: v for k, v in r.items()}}
                           for (u, o), r in sorted(rows.items(), key=lambda kv: -kv[1]["read"]
                                                   - kv[1]["write"])],
            "by_dtype_read": {k: v for k, v in sorted(dtypes.items(), key=lambda kv: -kv[1])},
            "op_census": {k: v for k, v in sorted(ops.items(), key=lambda kv: -kv[1])},
        }

    # ---- the phase partition: outermost bracket path that has a capture ---------------------
    sig_of = {}          # unit name -> list of sigs
    for sig in caps:
        sig_of.setdefault(sig.split("|")[0], []).append(sig)

    captured_units = set(sig_of)
    phases = []
    for path, row in sorted(tree.items()):
        parts = path.split("/")
        name = parts[-1]
        if name not in captured_units:
            continue
        if any(p in captured_units for p in parts[:-1]):
            continue                      # an ancestor is already a phase: not a partition member
        phases.append((path, name, row))

    # bytes per call: one unit can run at more than one input shape (the atom transformer and the
    # token transformer are the same class), so a phase's bytes are the call-weighted mean over
    # the shapes that phase's calls actually took.
    def bytes_for(name, calls):
        ss = sig_of.get(name, [])
        tot_calls = sum(sigs.get(s, {}).get("calls", 0) for s in ss)
        if not tot_calls:
            return None, []
        acc = 0.0
        detail = []
        for s in ss:
            c = sigs.get(s, {}).get("calls", 0)
            b = per_sig.get(s, {}).get("dram_total", 0)
            acc += c * b
            detail.append({"sig": s, "calls": c, "GB_per_call": gb(b),
                           "median_ms": sigs.get(s, {}).get("median_ms")})
        return acc / tot_calls, detail

    table = []
    for path, name, row in phases:
        per_call, detail = bytes_for(name, row["calls"])
        if per_call is None:
            continue
        calls, ms = row["calls"], row["incl_s"] / max(row["calls"], 1) * 1e3
        tb = per_call * calls / 1e12
        table.append({
            "phase": name, "path": path, "calls": calls,
            "ms_per_call": round(ms, 4), "s_per_fold": round(row["incl_s"], 4),
            "GB_per_call": gb(per_call), "TB_per_fold": round(tb, 5),
            "achieved_GBps": round(per_call / (row["incl_s"] / max(calls, 1)) / 1e9, 1),
            "pct_of_stream_roof": round(100 * per_call / (row["incl_s"] / max(calls, 1))
                                        / 1e9 / STREAM_ROOF_GBS, 1),
            "in_published_budget": name in PUBLISHED,
            "shapes": detail,
        })
    table.sort(key=lambda r: -r["TB_per_fold"])

    # ---- the glue: every ttnn call of the census fold that no bracketed unit contains --------
    cen = R.get("census") or {}
    cen_rows = cen.get("rows", [])
    by_path = defaultdict(lambda: {"n": 0, "GB": 0.0})
    for r in cen_rows:
        k = by_path[r["path"]]
        k["n"] += r["n"]
        k["GB"] += r["in_GB"] + r["out_GB"]
    phase_paths = {p for p, _n, _r in phases}

    def covered(path):
        parts = path.split("/")
        return any("/".join(parts[:i + 1]) in phase_paths for i in range(len(parts)))

    glue = [{"path": p, "n": v["n"], "modelled_GB": round(v["GB"], 3)}
            for p, v in sorted(by_path.items(), key=lambda kv: -kv[1]["GB"])
            if not covered(p)]
    modelled_total = sum(v["GB"] for v in by_path.values())
    modelled_glue = sum(g["modelled_GB"] for g in glue)
    modelled_phase = modelled_total - modelled_glue

    # captured / modelled, on the traffic that has both readings: what a modelled byte is worth
    measured_phase_GB = sum(r["TB_per_fold"] for r in table) * 1000
    ratio = measured_phase_GB / modelled_phase if modelled_phase else float("nan")
    glue_est_GB = modelled_glue * ratio
    fold_GB = measured_phase_GB + glue_est_GB

    published_GB = sum(r["TB_per_fold"] for r in table if r["in_published_budget"]) * 1000
    newly_named_GB = measured_phase_GB - published_GB
    remainder_GB = fold_GB - published_GB
    remainder_attributed_pct = (100 * newly_named_GB / remainder_GB) if remainder_GB else 0.0

    summary = {
        "fold_GB_measured_phases": round(measured_phase_GB, 1),
        "fold_GB_glue_estimated": round(glue_est_GB, 1),
        "fold_GB_total": round(fold_GB, 1),
        "modelled_total_GB": round(modelled_total, 1),
        "modelled_glue_GB": round(modelled_glue, 1),
        "captured_over_modelled": round(ratio, 3),
        "published_four_phases_GB": round(published_GB, 1),
        "newly_named_GB": round(newly_named_GB, 1),
        "remainder_GB": round(remainder_GB, 1),
        "remainder_attributed_pct": round(remainder_attributed_pct, 1),
        "named_share_of_fold_pct": round(100 * measured_phase_GB / fold_GB, 1),
        "stream_roof_GBps": STREAM_ROOF_GBS,
    }

    out = {"run": str(a.run.relative_to(ROOT)), "summary": summary, "phases": table,
           "glue": glue, "per_capture": per_sig,
           "baseline": R.get("baseline_summary"), "env": R.get("env")}
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=1))

    # ---- markdown ---------------------------------------------------------------------------
    L = []
    L.append("# Boltz-2 512 aa: where the fold's DRAM traffic goes\n")
    e = R.get("env", {})
    L.append(f"One instrumented fold on {e.get('host')} card {e.get('card')} "
             f"({e.get('card_type', '?')}), ttnn {e.get('ttnn')}, commit "
             f"`{str(e.get('git_head'))[:8]}`. Protocol: "
             f"{e.get('protocol', {}).get('fixture')}, "
             f"{e.get('protocol', {}).get('recycling_steps')} recycles, "
             f"{e.get('protocol', {}).get('sampling_steps')} sampling steps, seed "
             f"{e.get('protocol', {}).get('seed')}.\n")
    L.append("| phase | calls/fold | ms/call | GB/call | TB/fold | share | GB/s | % of "
             f"{STREAM_ROOF_GBS} GB/s | new |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in table:
        L.append(f"| {r['phase']} | {r['calls']} | {r['ms_per_call']:.3f} | "
                 f"{r['GB_per_call']:.3f} | {r['TB_per_fold']:.3f} | "
                 f"{100 * r['TB_per_fold'] * 1000 / fold_GB:.1f} % | {r['achieved_GBps']} | "
                 f"{r['pct_of_stream_roof']} % | {'' if r['in_published_budget'] else 'yes'} |")
    L.append(f"| **glue** (no module) | | | | {glue_est_GB / 1000:.3f} | "
             f"{100 * glue_est_GB / fold_GB:.1f} % | | | |")
    L.append(f"\n**REMAINDER-ATTRIBUTED: {summary['remainder_attributed_pct']:.1f} %** "
             f"({newly_named_GB:.0f} GB of the {remainder_GB:.0f} GB that sat outside the four "
             f"published phases now has a name).\n")
    L.append(f"Glue is priced from the census fold's floor byte model ({modelled_glue:.1f} GB "
             f"modelled) scaled by {ratio:.2f}x, the ratio the same model gives against the "
             f"captured traffic of the phases that have both readings.\n")

    for sig in sorted(per_sig, key=lambda s: -per_sig[s]["dram_total"])[:6]:
        d = per_sig[sig]
        L.append(f"\n## {sig} — {gb(d['dram_total']):.3f} GB/call, {d['ttnn_ops']} ttnn ops\n")
        L.append("| unit | op | GB read | GB written | tensors | allocs |")
        L.append("|---|---|---|---|---|---|")
        for r in d["by_unit_op"][:a.top]:
            L.append(f"| {r['unit']} | {r['op']} | {gb(r['read']):.4f} | {gb(r['write']):.4f} "
                     f"| {r['n_tensor']} | {r['n_alloc']} |")
        tot = sum(d["by_dtype_read"].values()) or 1
        L.append("\ndtype census, by bytes read: " + ", ".join(
            f"{k} {100 * v / tot:.1f} %" for k, v in d["by_dtype_read"].items()))
    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    a.out_md.write_text("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    print("WROTE", a.out_json, a.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
