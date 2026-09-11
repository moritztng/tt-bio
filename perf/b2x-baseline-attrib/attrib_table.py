#!/usr/bin/env python3
"""Turn one instrumented fold into the per-phase, per-unit, per-op and per-dtype attribution.

Host only, no device. Reads what ``baseline_attrib.py`` wrote:

  * ``attrib.tree``   per bracket path: calls, inclusive and exclusive wall
  * ``attrib.sigs``   per (unit, input shape): calls, median ms
  * ``captures/``     one ``ttnn.graph`` capture per (unit, input shape)
  * ``census.rows``   every ttnn call of a second fold, charged to the bracket stack under a
                      floor byte model: the only reading of the ops that belong to no module

Byte accounting inside a capture is the same as ``perf/bioir_roofline/fold_bytes_512.py`` -- a
DRAM-resident tensor read once per region, a DRAM buffer allocated inside the region written
once -- so a phase's GB/call is directly comparable to the published byte budget. What is new is
where each byte is charged: to the innermost ttnn op and the innermost bracketed unit that
encloses it. Region totals are unchanged by that, and the table prints them so it can be checked.

Two statistics, deliberately not mixed. **Bytes** come from the capture and are exact. **ms/call**
is the median over that unit's calls in the bracketed fold, the same statistic the published
budget used; nested brackets each carry a device sync, which inflates the small units by 10-20 %,
so every GB/s here is a floor.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Measured on this card by perf/bioir_roofline/roofs_bh.py (roofs_p300c_qb2_card2.json).
STREAM_ROOF_GBS = 429.9
# One bf16 pair tensor at 512 tokens, the unit the campaign counts trunk traffic in.
Z_BYTES = 512 * 512 * 128 * 2

# What the published byte budget (state/bioir-roofline/FINDINGS.md) instrumented, by capture
# signature, so the table can say which rows are new.
PUBLISHED_SIGS = {
    "PairformerLayer|1x512x384,1x512x512x128",
    "PairformerLayer|1x512x512x128",
    "DiffusionTransformerLayer|1x512x768,1x512x768",
    "DiffusionTransformerLayer|1x224x32x128,1x224x32x128",
}


def walk(nodes):
    """Charge every DRAM tensor read and DRAM allocation to (unit, op), in capture order."""
    stack: list[str] = []
    seen: set[int] = set()
    rows: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"read": 0, "write": 0, "l1": 0, "n_tensor": 0, "n_alloc": 0})
    dtypes: dict[str, int] = defaultdict(int)
    ops: dict[str, int] = defaultdict(int)
    units: dict[str, dict] = defaultdict(lambda: {"read": 0, "write": 0, "l1": 0, "ops": 0})
    total = {"read": 0, "write": 0, "l1": 0, "ops": 0}

    def here():
        unit = "(self)"
        for f in reversed(stack):
            if f.startswith("unit::"):
                unit = f[6:]
                break
        op, started = None, False
        for f in stack:
            if f.startswith("unit::"):
                started, op = True, None
                continue
            if started and op is None and (f.startswith("ttnn.") or f.startswith("ttnn::")):
                op = f
        return unit, (op or (stack[-1] if stack else "(none)"))

    for n in nodes:
        t = n.get("node_type")
        p = n.get("params") or {}
        if t == "function_start":
            name = str(p.get("name", ""))
            if name.startswith("ttnn."):
                total["ops"] += 1
                ops[name] += 1
                u, _o = here()
                units[u]["ops"] += 1
            stack.append(name)
        elif t == "function_end":
            if stack:
                stack.pop()
        elif t in ("tensor", "buffer_allocate"):
            u, o = here()
            r, ur = rows[(u, o)], units[u]
            if t == "tensor":
                tid = p.get("tensor_id")
                if tid in seen or "DRAM" not in str(p.get("buffer_type", "")):
                    continue
                seen.add(tid)
                sz = int(p.get("size", 0) or 0)
                r["read"] += sz
                r["n_tensor"] += 1
                ur["read"] += sz
                dtypes[str(p.get("dtype", "?")).split("::")[-1]] += sz
                total["read"] += sz
            else:
                sz = int(p.get("size", 0) or 0)
                if str(p.get("type")) == "DRAM":
                    r["write"] += sz
                    r["n_alloc"] += 1
                    ur["write"] += sz
                    total["write"] += sz
                else:
                    r["l1"] += sz
                    ur["l1"] += sz
                    total["l1"] += sz
    return rows, dtypes, ops, units, total


def gb(x):
    return round(x / 1e9, 6)


def assign_sigs(name, part_paths, excl_paths, sigs):
    """Which capture signature belongs to which tree path of the same unit class.

    One class runs at more than one call site: ``PairformerLayer`` is the trunk's block, the
    confidence head's block and the block inside an ``MSALayer``, and only the first two are
    phases. Signatures carry no call site, so they are matched on call count -- exact match
    first, and whatever is left is shared by the remaining paths, which is only ever right
    because the sharing paths run the identical shape (trunk and confidence blocks do).
    """
    free = {s: c for s, c in sigs.items()}
    out = {}
    for path, calls in list(excl_paths.items()) + list(part_paths.items()):
        hit = next((s for s, c in free.items() if c == calls), None)
        if hit is not None:
            out[path] = [hit]
            del free[hit]
    rest = [p for p in list(excl_paths) + list(part_paths) if p not in out]
    for p in rest:
        out[p] = list(free)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--baseline-s", type=float, default=None,
                    help="the plain-arm median this fold's phases are read against")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--min-section-gb", type=float, default=0.1,
                    help="smallest per-call unit that gets its own section. The diffusion "
                         "token layer moves 0.214 GB/call and is what b2x-diffusion-layer-"
                         "bytes is priced off, so this cannot sit at 1 GB.")
    a = ap.parse_args()

    R = json.loads(a.run.read_text())
    at = R["attrib"]
    tree, sigs, caps = at["tree"], at["sigs"], at["captures"]
    base_s = a.baseline_s or (R.get("baseline_summary") or {}).get("plain_median_s")

    # ---- parse every capture ---------------------------------------------------------------
    per_sig = {}
    for sig, info in sorted(caps.items()):
        f = ROOT / info["file"] if not Path(info["file"]).is_absolute() else Path(info["file"])
        if not f.exists():
            f = Path(str(a.run.parent / Path(info["file"]).name))
        if not f.exists():
            continue
        with gzip.open(f, "rt") as fh:
            nodes = json.load(fh)
        rows, dtypes, ops, units, total = walk(nodes)
        per_sig[sig] = {
            "nodes": len(nodes), "ttnn_ops": total["ops"],
            "dram_read": total["read"], "dram_write": total["write"],
            "dram_total": total["read"] + total["write"], "l1_alloc": total["l1"],
            "GB_per_call": gb(total["read"] + total["write"]),
            "Z_per_call": round((total["read"] + total["write"]) / Z_BYTES, 1),
            "by_unit": {u: {k: gb(v) if k in ("read", "write", "l1") else v
                            for k, v in d.items()}
                        for u, d in sorted(units.items(), key=lambda kv: -kv[1]["read"])},
            "by_unit_op": [{"unit": u, "op": o, **r}
                           for (u, o), r in sorted(rows.items(),
                                                   key=lambda kv: -kv[1]["read"] - kv[1]["write"])],
            "by_dtype_read_bytes": dict(sorted(dtypes.items(), key=lambda kv: -kv[1])),
            "op_census": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
        }

    # ---- the phase partition ---------------------------------------------------------------
    captured = {s.split("|")[0] for s in caps}
    part, excl = defaultdict(dict), defaultdict(dict)
    for path, row in tree.items():
        parts = path.split("/")
        name = parts[-1]
        if name not in captured:
            continue
        (excl if any(p in captured for p in parts[:-1]) else part)[name][path] = row["calls"]

    sig_calls = defaultdict(dict)
    for s, r in sigs.items():
        if s.split("|")[0] in captured:
            sig_calls[s.split("|")[0]][s] = r["calls"]

    table = []
    for name in sorted(part):
        amap = assign_sigs(name, part[name], excl.get(name, {}), sig_calls[name])
        for path, calls in part[name].items():
            ss = [s for s in amap.get(path, []) if s in per_sig]
            if not ss:
                continue
            w = sum(sigs[s]["calls"] for s in ss)
            per_call = sum(sigs[s]["calls"] * per_sig[s]["dram_total"] for s in ss) / w
            ms = sum(sigs[s]["calls"] * sigs[s]["median_ms"] for s in ss) / w
            table.append({
                "phase": path, "unit": name, "calls": calls,
                "ms_per_call": round(ms, 4), "s_per_fold": round(ms * calls / 1e3, 3),
                "GB_per_call": gb(per_call), "TB_per_fold": round(per_call * calls / 1e12, 5),
                "Z_per_call": round(per_call / Z_BYTES, 1),
                "achieved_GBps": round(per_call / (ms / 1e3) / 1e9, 1),
                "pct_of_stream_roof": round(100 * per_call / (ms / 1e3) / 1e9
                                            / STREAM_ROOF_GBS, 1),
                "sigs": ss,
                "in_published_budget": all(s in PUBLISHED_SIGS for s in ss),
            })
    table.sort(key=lambda r: -r["TB_per_fold"])

    # ---- the glue: ttnn calls of the census fold that no phase contains ----------------------
    by_path = defaultdict(lambda: {"n": 0, "GB": 0.0})
    for r in (R.get("census") or {}).get("rows", []):
        k = by_path[r["path"]]
        k["n"] += r["n"]
        k["GB"] += r["in_GB"] + r["out_GB"]
    phase_paths = {r["phase"] for r in table}

    def covered(path):
        parts = path.split("/")
        return any("/".join(parts[:i + 1]) in phase_paths for i in range(len(parts)))

    glue = [{"path": p, "n": v["n"], "modelled_GB": round(v["GB"], 3)}
            for p, v in sorted(by_path.items(), key=lambda kv: -kv[1]["GB"]) if not covered(p)]
    modelled_total = sum(v["GB"] for v in by_path.values())
    modelled_glue = sum(g["modelled_GB"] for g in glue)
    measured_phase_GB = sum(r["TB_per_fold"] for r in table) * 1000
    ratio = measured_phase_GB / (modelled_total - modelled_glue) if modelled_total else 1.0
    glue_GB = modelled_glue * ratio
    fold_GB = measured_phase_GB + glue_GB

    published_GB = sum(r["TB_per_fold"] for r in table if r["in_published_budget"]) * 1000
    # The published budget's four phases as they were counted there: the two pairformer
    # signatures and the two diffusion layer signatures. Everything else in this fold -- the
    # rest of an MSALayer, the diffusion conditioning, the glue -- was its 1.400 TB remainder.
    pub_rows = []
    for s in PUBLISHED_SIGS:
        if s in per_sig and s in sigs:
            pub_rows.append({"sig": s, "calls": sigs[s]["calls"],
                             "GB_per_call": per_sig[s]["GB_per_call"],
                             "median_ms": sigs[s]["median_ms"],
                             "TB_per_fold": round(per_sig[s]["dram_total"]
                                                  * sigs[s]["calls"] / 1e12, 5)})
    pub_total_GB = sum(r["TB_per_fold"] for r in pub_rows) * 1000
    remainder_GB = fold_GB - pub_total_GB
    named_remainder_GB = measured_phase_GB - pub_total_GB

    phase_s = sum(r["s_per_fold"] for r in table)
    summary = {
        "fold_GB_named_phases": round(measured_phase_GB, 1),
        "fold_GB_glue": round(glue_GB, 1),
        "fold_GB_total": round(fold_GB, 1),
        "fold_TB_total": round(fold_GB / 1000, 3),
        "published_estimate_TB": 6.45,
        "modelled_total_GB": round(modelled_total, 1),
        "modelled_glue_GB": round(modelled_glue, 1),
        "captured_over_modelled": round(ratio, 3),
        "published_four_sigs_GB": round(pub_total_GB, 1),
        "remainder_GB": round(remainder_GB, 1),
        "remainder_named_GB": round(named_remainder_GB, 1),
        "remainder_attributed_pct": round(100 * named_remainder_GB / remainder_GB, 1),
        "named_share_of_fold_pct": round(100 * measured_phase_GB / fold_GB, 1),
        "phase_s_of_fold": round(phase_s, 2),
        "baseline_fold_s": base_s,
        "phase_time_share_pct": round(100 * phase_s / base_s, 1) if base_s else None,
        "achieved_GBps_on_baseline": round(fold_GB / base_s, 1) if base_s else None,
        "stream_roof_GBps": STREAM_ROOF_GBS,
    }

    out = {"run": str(a.run), "summary": summary, "phases": table,
           "published_four_sigs": pub_rows, "glue": glue, "per_capture": per_sig,
           "baseline": R.get("baseline_summary"), "control_298": R.get("control_298"),
           "env": R.get("env")}
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=1))

    # ---- markdown ----------------------------------------------------------------------------
    e = R.get("env", {})
    L = ["# Boltz-2 512 aa: every byte the fold moves, by module\n"]
    L.append(f"One instrumented fold on {e.get('host')} card {e.get('card')} "
             f"({e.get('card_type', 'p300c')}), ttnn {e.get('ttnn')}, commit "
             f"`{str(e.get('git_head'))[:8]}`, "
             f"{e.get('protocol', {}).get('recycling_steps')} recycles / "
             f"{e.get('protocol', {}).get('sampling_steps')} sampling steps / seed "
             f"{e.get('protocol', {}).get('seed')} on "
             f"`perf/size512/fixtures/cdk2x2_512.yaml`.\n")
    L.append(f"The fold moves **{fold_GB / 1000:.3f} TB**, not the 6.45 TB the published budget "
             f"estimated. {summary['named_share_of_fold_pct']:.1f} % of it is inside a named "
             f"module; the ops that belong to no module move {glue_GB:.1f} GB.\n")
    L.append("| phase | calls | ms/call | GB/call | Z/call | TB/fold | share | GB/s | % of roof |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in table:
        L.append(f"| {r['phase']} | {r['calls']} | {r['ms_per_call']:.3f} | "
                 f"{r['GB_per_call']:.3f} | {r['Z_per_call']} | {r['TB_per_fold']:.3f} | "
                 f"{100 * r['TB_per_fold'] * 1000 / fold_GB:.1f} % | {r['achieved_GBps']} | "
                 f"{r['pct_of_stream_roof']} % |")
    L.append(f"| glue (no module) | | | | | {glue_GB / 1000:.3f} | "
             f"{100 * glue_GB / fold_GB:.1f} % | | |")
    L.append(f"\nZ is one bf16 pair tensor at 512 tokens, {Z_BYTES / 1e6:.2f} MB. ms/call is the "
             "median over that unit's calls; every nested bracket carries a device sync, which "
             "inflates the smaller units, so each GB/s here is a floor.\n")
    L.append(f"**REMAINDER-ATTRIBUTED: {summary['remainder_attributed_pct']:.1f} %** — the four "
             f"signatures the published budget instrumented carry {pub_total_GB:.0f} GB, leaving "
             f"{remainder_GB:.0f} GB outside them, of which {named_remainder_GB:.0f} GB now sits "
             f"in a named module.\n")

    for sig in sorted(per_sig, key=lambda s: -per_sig[s]["dram_total"]):
        d = per_sig[sig]
        if d["dram_total"] < a.min_section_gb * 1e9:
            continue
        L.append(f"\n## {sig} — {d['GB_per_call']:.3f} GB/call, {d['Z_per_call']} Z, "
                 f"{d['ttnn_ops']} ttnn ops\n")
        L.append("| sub-unit | GB read | GB written | ttnn ops |")
        L.append("|---|---|---|---|")
        for u, v in d["by_unit"].items():
            L.append(f"| {u} | {v['read']:.4f} | {v['write']:.4f} | {v['ops']} |")
        L.append("\n| sub-unit | op | GB read | GB written |")
        L.append("|---|---|---|---|")
        for r in d["by_unit_op"][:a.top]:
            L.append(f"| {r['unit']} | {r['op']} | {gb(r['read']):.4f} | {gb(r['write']):.4f} |")
        tot = sum(d["by_dtype_read_bytes"].values()) or 1
        L.append("\ndtype, by DRAM bytes read: " + ", ".join(
            f"{k} {100 * v / tot:.2f} % ({v / 1e6:.1f} MB)"
            for k, v in d["by_dtype_read_bytes"].items()))
    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    a.out_md.write_text("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    print("WROTE", a.out_json, a.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
