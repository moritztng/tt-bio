#!/usr/bin/env python3
"""Score every arm on one scope with one function and apply PREDICTION.md's decision rule.

Both sides are scored by amp_arm.score_grads over dumped tensors: ours from msa_instrument.py's
--dump-grads, upstream from msa_amp_arm.py's --dump. The matched scope is the 151 names our arm
scores. Writes AMPLIFICATION.json.
"""
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_tapeamp"))
from amp_arm import score_grads  # noqa: E402

S = Path("/home/ttuser/of3t_msaamp")
B = torch.load(S / "cap043b/boundary_msa_module.pt", map_location="cpu", weights_only=False)
REF = B["param_grads"]
ours_rep = json.loads((HERE / "ours.json").read_text())
MATCH = {r["name"] for r in ours_rep["gradient"]["rows"]}
UP = ["bf16auto", "bf16auto_AA", "f32", "f64", "bf16in", "bf16auto_BREAK"]


def load(tag):
    d = torch.load(S / f"grads_{tag}.pt", map_location="cpu", weights_only=False)
    return d["grads"] if isinstance(d, dict) and "grads" in d else d


def geometry(g, names):
    """rel, r = |g|/|ref|, cos on the concatenated scope, with A43's identity residual."""
    ks = sorted(names)
    a = torch.cat([g[k].reshape(-1).double() for k in ks])
    b = torch.cat([REF[k].reshape(-1).double() for k in ks])
    rel = float((a - b).norm() / b.norm())
    r = float(a.norm() / b.norm())
    cos = float(a @ b / (a.norm() * b.norm()))
    return {"rel": rel, "r": r, "cos": cos,
            "identity_residual": rel ** 2 - (1 + r ** 2 - 2 * r * cos)}


def family(name):
    m = re.match(r"msa_module\.blocks\.(\d+)\.(?:pair_stack\.)?([a-z_]+)", name)
    return (int(m.group(1)), m.group(2)) if m else (None, name)


def grouped(per_o, per_u, key):
    out = {}
    for nm in per_o:
        k = key(nm)
        e = out.setdefault(str(k), {"n": 0, "err_o": 0.0, "err_u": 0.0, "ref": 0.0})
        s = float((REF[nm].double() ** 2).sum())
        e["n"] += 1
        e["ref"] += s
        e["err_o"] += per_o[nm] ** 2 * s
        e["err_u"] += per_u[nm] ** 2 * s
    for e in out.values():
        e["G_ours"] = (e["err_o"] / e["ref"]) ** 0.5
        e["G_upstream_bf16"] = (e["err_u"] / e["ref"]) ** 0.5
        e["ours_over_upstream"] = e["G_ours"] / e["G_upstream_bf16"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["err_o"]))


rep = {"matched_scope_n": len(MATCH), "arms": {}}
G = {"ours": load("ours")}
F = {"ours": ours_rep["forward"]["rel_l2_real_block"]}
for t in UP:
    G[t] = load(t)
    F[t] = json.loads((HERE / f"AMP_{t}.json").read_text())["FORWARD"]["real_block"]
PER = {}
for t, g in G.items():
    m, per = score_grads(g, REF, names=MATCH)
    full = score_grads(g, REF)[0] if t != "ours" else None
    PER[t] = per
    ent = {"forward_real_block": F[t], "matched": m, "full_scope": full}
    if F[t]:
        ent["ratio_matched"] = {"mass_weighted": m["mass_weighted_rel"] / F[t],
                                "median": m["median_rel"] / F[t]}
    if t in ("ours", "bf16auto", "f32"):
        ent["geometry_matched"] = geometry(g, MATCH)
    rep["arms"][t] = ent

o, u = rep["arms"]["ours"], rep["arms"]["bf16auto"]
dec = {}
for stat, key in (("mass_weighted", "mass_weighted_rel"), ("median", "median_rel")):
    Go, Gu = o["matched"][key], u["matched"][key]
    Ro, Ru = o["ratio_matched"][stat], u["ratio_matched"][stat]
    br = ("B3" if Go > 1.25 * Gu else "B1" if Ru >= 0.5 * Ro else "B2")
    d = {"G_ours": Go, "G_upstream_bf16": Gu, "G_ours_over_upstream": Go / Gu,
         "F_ours": o["forward_real_block"], "F_upstream_bf16": u["forward_real_block"],
         "F_ours_over_upstream": o["forward_real_block"] / u["forward_real_block"],
         "R_ours": Ro, "R_upstream_bf16": Ru, "R_ours_over_upstream": Ro / Ru,
         "branch": br, "B4_no_conditioning": Ru < 2.0}
    if br == "B3":
        Gin = rep["arms"]["bf16in"]["matched"][key]
        worst3 = sorted(PER["ours"], key=lambda n: -PER["ours"][n])[:3]
        keep = MATCH - set(worst3)
        o3 = score_grads(G["ours"], REF, names=keep)[0][key]
        u3 = score_grads(G["bf16auto"], REF, names=keep)[0][key]
        d["B3_exclusions"] = {
            "i_input_precision": {"G_upstream_bf16_inputs": Gin, "threshold": Go / 1.25,
                                  "carries_it": Gin >= Go / 1.25},
            "ii_tail": {"dropped": worst3, "G_ours": o3, "G_upstream_bf16": u3,
                        "ratio": o3 / u3, "carries_it": o3 <= 1.25 * u3},
            "iii_forward": {"F_ours_over_upstream": d["F_ours_over_upstream"],
                            "forward_also_worse": d["F_ours_over_upstream"] > 1.25,
                            "backward_own_share_R_ours_over_R_upstream": Ro / Ru},
        }
    dec[stat] = d
rep["DECISION"] = dec
rep["BY_FAMILY_mass_weighted"] = grouped(PER["ours"], PER["bf16auto"], lambda n: family(n)[1])
rep["BY_BLOCK_mass_weighted"] = grouped(PER["ours"], PER["bf16auto"], lambda n: family(n)[0])
rep["TOP_TENSORS_by_error_mass_ours"] = sorted(
    ({"name": n, "rel_ours": PER["ours"][n], "rel_upstream_bf16": PER["bf16auto"][n],
      "share_of_ours_error_mass": PER["ours"][n] ** 2 * float((REF[n].double() ** 2).sum())
      / o["matched"]["error_mass"]} for n in MATCH), key=lambda e: -e["share_of_ours_error_mass"])[:12]
aa = [json.loads((HERE / f"AMP_{t}.json").read_text()) for t in ("bf16auto", "bf16auto_AA")]
rep["AA_bit_identical"] = {
    "forward": aa[0]["FORWARD"] == aa[1]["FORWARD"],
    "per_tensor": aa[0]["PER_TENSOR"] == aa[1]["PER_TENSOR"],
    "ours_forward": json.loads((HERE / "ours.json").read_text())["forward"]
    == json.loads((HERE / "ours_AA.json").read_text())["forward"],
    "ours_per_tensor": sorted((r["name"], r["rel_l2"]) for r in ours_rep["gradient"]["rows"])
    == sorted((r["name"], r["rel_l2"]) for r in
              json.loads((HERE / "ours_AA.json").read_text())["gradient"]["rows"])}
(HERE / "AMPLIFICATION.json").write_text(json.dumps(rep, indent=1) + "\n")
for t, e in rep["arms"].items():
    m = e["matched"]
    print(f"{t:15s} F {e['forward_real_block']:.6e}  G_mw {m['mass_weighted_rel']:.6e}  "
          f"G_med {m['median_rel']:.6e}  R {e.get('ratio_matched')}")
print(json.dumps(rep["DECISION"], indent=1))
print(json.dumps(rep["AA_bit_identical"]))
for k, v in list(rep["BY_FAMILY_mass_weighted"].items())[:8]:
    print(f"  {k:22s} n={v['n']:3d} G_o {v['G_ours']:.4e} G_u {v['G_upstream_bf16']:.4e} "
          f"x{v['ours_over_upstream']:.2f} err_share {v['err_o']/o['matched']['error_mass']:.3f}")
for t in ("ours", "bf16auto", "f32"):
    print(t, rep["arms"][t]["geometry_matched"])
