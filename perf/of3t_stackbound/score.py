#!/usr/bin/env python3
"""of3t-stackbound: every trunk arm on the 4hhb frame, scored on THIS boundary's bar.

  score.py --arm TAG=dev_TAG.pt ... --out LADDER_SCORES.json

What is native to this boundary: the bar (A26 rule over the 3,660-tensor union, from this
boundary's bf16 and float64 references), every section's reference mass, and the trunk reading
of every arm. What is not: the five non-trunk scopes' readings, held at stackexact's banked
graded values (R1) because their device arms exist only on 5nw3. Every clause figure carries
that declaration, and R0 (non-trunk exact) is printed beside it. PREREGISTERED.md fixes all of
this before any arm ran.

The per-tensor rows are `perf/of3t_trajectory/agreement.py`'s, the composition and the
allowance are `perf/of3t_modelframe/clause.py`'s `recompose` and `solve`, imported. The
recomposition CONTROL runs first: it must reproduce all four of stackexact's published clause
values from their own section tables, or nothing below is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
sys.path.insert(0, str(PERF / "of3t_trajectory"))
sys.path.insert(0, str(PERF / "of3t_modelframe"))
import agreement  # noqa: E402
from clause import recompose, solve  # noqa: E402

SEC = "pairformer_stack"
O = Path("/home/ttuser/of3t_stackbound")
BANKED = {t: PERF / "of3t_stackexact" / f"MODEL_{t}_composed3660_n384.json"
          for t in ("SHIP_A", "S", "L", "SL")}
BANKED_CLAUSE = {"SHIP_A": 0.22072451195864032, "S": 0.1983072658753951,
                 "L": 0.19527710571784584, "SL": 0.14940227528507738}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()


def load(p, names, key=None):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if key and isinstance(d, dict) and key in d:
        d = d[key]
    out = {n: (d[n].to(torch.float64).reshape(-1) if d.get(n) is not None else None)
           for n in names}
    del d
    return out


def triple(st):
    """The concatenated (rel, r, cos) with its A43 identity residual and the angle."""
    rel, r, c = st["mass_weighted_rel_l2"], st["mass_weighted_norm_ratio"], st["mass_weighted_cos"]
    return {"rel": rel, "r": r, "cos": c,
            "angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, c)))),
            "A43_residual": abs(rel * rel - (1 + r * r - 2 * r * c))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    # ---- CONTROL: the composition identity on stackexact's own artifacts ------------------
    control = {}
    for t, p in BANKED.items():
        d = json.loads(p.read_text())
        got = recompose(d["per_section"]["renorm_vs_UPSTREAM_BF16"])
        control[t] = {"published": BANKED_CLAUSE[t], "recomposed": got,
                      "rel_difference": abs(got - BANKED_CLAUSE[t]) / BANKED_CLAUSE[t]}
        if control[t]["rel_difference"] > 1e-15:
            raise SystemExit(f"CONTROL fails on {t}: {control[t]} -- nothing below may be read")
    banked_rest = {s: v["mass_weighted_rel_l2"] for s, v in json.loads(
        BANKED["SHIP_A"].read_text())["per_section"]["renorm_vs_UPSTREAM_BF16"].items() if s != SEC}

    union = json.loads((HERE / "UNION3660.json").read_text())["names"]
    sections = list(json.loads((PERF / "of3t_orchestrator" / "SECTION_MASS_MEASURED.json")
                               .read_text())["sections_pct_of_model"])
    refs = {"float64": O / "ref_f64" / "grads_f64.pt", "upstream_bf16": O / "ref_bf16" / "grads_f64.pt",
            "upstream_f32": O / "ref_f32" / "grads_f64.pt"}
    inputs = {k: {"path": str(p), "sha256": sha(p)} for k, p in refs.items()}

    full = torch.load(refs["float64"], map_location="cpu", weights_only=False)
    total_sq = sum(float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
                   for v in full.values() if v is not None)
    missing = [n for n in union if full.get(n) is None]
    del full
    agreement.MODEL_TOTAL_SQ = total_sq
    f64 = load(refs["float64"], union)
    bf16 = load(refs["upstream_bf16"], union)
    f32 = load(refs["upstream_f32"], union)

    def bysec(rows):
        g = {}
        for r in rows:
            g.setdefault(r["section"], []).append(r)
        return {s: dict(agreement.stat(rs, s), ref_sq=sum(r["ref_norm"] ** 2 for r in rs))
                for s, rs in sorted(g.items())}

    floor_rows = agreement.pair_rows(f64, bf16, union, f64, sections)
    floor = agreement.stat(floor_rows, "UPSTREAM_BF16_vs_FLOAT64")
    f32_floor = agreement.stat(agreement.pair_rows(f64, f32, union, f64, sections),
                               "UPSTREAM_F32_vs_FLOAT64")
    # the graded composition weights every section by ITS bf16 reference mass, as model_scope does
    ref_self = agreement.pair_rows(bf16, bf16, union, f64, sections)
    ref_sq = {s: v["ref_sq"] for s, v in bysec(ref_self).items()}
    bar = math.sqrt(2.0) * floor["mass_weighted_rel_l2"] / floor["mass_weighted_norm_ratio"]
    floor_sec = bysec(floor_rows)
    rest_secs = sorted(s for s in ref_sq if s != SEC)
    assert set(rest_secs) == set(banked_rest), (rest_secs, sorted(banked_rest))

    def table(trunk_rel, rest):
        t = {s: {"ref_sq": ref_sq[s], "mass_weighted_rel_l2": rest[s]} for s in rest_secs}
        t[SEC] = {"ref_sq": ref_sq[SEC], "mass_weighted_rel_l2": trunk_rel}
        return t

    R1 = banked_rest
    R0 = {s: 0.0 for s in rest_secs}
    trunk_names = [n for n in union if n.startswith(SEC + ".")]
    out = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
           "row": "of3t-stackbound", "device_involved": False,
           "why_no_aiclk": "CPU only; each arm's device host, board, card and DURING AICLK are in "
                           "its DEV_<tag>.json, stamped by arm.sh",
           "boundary_version": "upstream OpenFold3 0.4.3, 4hhb boundary, 384 real tokens of 384",
           "CONTROL": control, "inputs": inputs,
           "union": {"n": len(union), "n_absent_from_float64": len(missing)},
           "model_squared_gradient_norm": total_sq,
           "bar": {"rule": "A26: sqrt(2) * floor / r over the union, this boundary only",
                   "upstream_own_bf16_floor_vs_float64": floor["mass_weighted_rel_l2"],
                   "norm_ratio_bf16_over_float64": floor["mass_weighted_norm_ratio"],
                   "A26_reachable_bar_vs_their_bf16": bar,
                   "instrument_floor_upstream_f32_vs_float64": f32_floor["mass_weighted_rel_l2"]},
           "sections": {s: {"ref_sq_bf16": ref_sq[s],
                            "pct_of_model_mass_float64": floor_sec[s]["pct_of_model_mass"],
                            "upstream_bf16_floor_vs_float64": floor_sec[s]["mass_weighted_rel_l2"],
                            "rest_R1_transported_from_5nw3": R1.get(s)} for s in ref_sq},
           "rest_declaration": "R1: the ten non-trunk sections at of3t-stackexact's banked graded "
                               "readings on the 5nw3 boundary (MODEL_SHIP_A per_section), weighted "
                               "by this boundary's reference mass. R0: non-trunk exact.",
           "allowance": {"R1": solve(table(0.0, R1), bar), "R0": solve(table(0.0, R0), bar)},
           "levels": {"bit_exact_float64_trunk_R1": recompose(table(0.0, R1)) / bar,
                      "upstreams_own_floor_trunk_R1":
                          recompose(table(floor_sec[SEC]["mass_weighted_rel_l2"], R1)) / bar},
           "arms": {}}
    for spec in a.arm:
        tag, p = spec.split("=", 1)
        arm = load(p, trunk_names, key="grads")
        g = agreement.stat(agreement.pair_rows(bf16, arm, trunk_names, f64, sections), "graded")
        c = agreement.stat(agreement.pair_rows(f64, arm, trunk_names, f64, sections), "float64")
        t = g["mass_weighted_rel_l2"]
        out["arms"][tag] = {
            "pt": {"path": p, "sha256": sha(p)},
            "trunk_vs_upstream_bf16": dict(triple(g), worst_tensor=g["worst_tensor"],
                                           worst_rel_l2=g["worst_rel_l2"], n=g["n"]),
            "trunk_vs_float64": dict(triple(c), worst_tensor=c["worst_tensor"],
                                     worst_rel_l2=c["worst_rel_l2"], n=c["n"]),
            "trunk_over_allowance_R1": t / out["allowance"]["R1"],
            "clause_R1": recompose(table(t, R1)), "x_bar_R1": recompose(table(t, R1)) / bar,
            "clause_R0": recompose(table(t, R0)), "x_bar_R0": recompose(table(t, R0)) / bar,
        }
        del arm
        print(tag, json.dumps({k: out["arms"][tag][k] for k in ("x_bar_R1", "x_bar_R0")}), flush=True)
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("CONTROL", "bar", "allowance", "levels")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
