#!/usr/bin/env python3
"""Render this row's state doc from the artifacts, so no number is retyped.

Every figure in `state/of3t-bondcov.md` comes out of a json written by a run. The
template below carries the prose; the substitutions carry the evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _headrow(rows, pdb_id, datapoint):
    for r in rows:
        if r["pdb_id"] == pdb_id and r["datapoint"] == datapoint:
            return r
    raise SystemExit(f"{pdb_id} {datapoint} is not in the mask sweep")


def _partners(rows, pdb_id, datapoint):
    """The mask's own entries, spelled out, so no index is typed into the prose."""
    out = []
    for p in _headrow(rows, pdb_id, datapoint)["partners"]:
        out.append(f"(i = {p['i']}, `is_ligand` {p['i_is_ligand']}, `is_polymer` "
                   f"{p['i_is_polymer']}; j = {p['j']}, `is_polymer` {p['j_is_polymer']}, "
                   f"`is_ligand` {p['j_is_ligand']})")
    return "; ".join(out) if out else "(none)"


def fmt(x, n=6):
    if x is None:
        return "None"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{n}e}" if (x != 0 and abs(x) < 1e-3) else f"{x:.{n}g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, type=Path,
                    help="the headline mask sweep")
    ap.add_argument("--mask2", type=Path,
                    help="a second sweep at another crop, to show the crop is a draw")
    ap.add_argument("--grad", required=True, type=Path, nargs="+",
                    help="bond_coverage.py reports; the first is the headline target")
    ap.add_argument("--baseline", required=True, type=Path,
                    help="of3t-auxheads' zero reading on the 8-structure corpus")
    ap.add_argument("--evidence", required=True, type=Path,
                    help="the evidence directory the coverage table reads")
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    mask = json.loads(a.mask.read_text())
    mask2 = json.loads(a.mask2.read_text()) if a.mask2 else None
    grads = [json.loads(p.read_text()) for p in a.grad]
    base = json.loads(a.baseline.read_text())

    rows = mask["rows"]
    by_pdb: dict[str, list] = {}
    for r in rows:
        by_pdb.setdefault(r["pdb_id"], []).append(r)

    mask_table = "\n".join(
        f"| {r['pdb_id']} | {r['datapoint']} | {r['index']} | {r['n_polymer_tokens']} | "
        f"{r['n_ligand_tokens']} | {r['token_bonds_nnz']} | "
        f"{r['polymer_ligand_pairs_either_orientation']} | **{r['bond_mask_nnz']}** |"
        for r in rows)

    def gline(g):
        c = g["bond_gradient_contribution"]
        t = g["target"]
        return (f"| {t['pdb_id']} {t['datapoint']} | {t['bond_mask_nnz']} | "
                f"{g['arms']['bond4']['breakdown'].get('bond_loss')!r} | "
                f"{fmt(c['squared_norm_of_difference'])} | "
                f"{fmt(c['squared_norm_with_bond'])} | "
                f"{fmt(c['share_of_squared_gradient_norm'])} | "
                f"{c['n_params_moved']:,} of {c['n_params']:,} |")

    bc = base["bond_gradient_contribution"]
    grad_table = "\n".join(
        [f"| 5nw3 (of3t-auxheads) | 0 | {base['arms']['bond4']['breakdown'].get('bond_loss')!r} | "
         f"{fmt(bc['squared_norm_of_difference'])} | {fmt(bc['squared_norm_with_bond'])} | "
         f"{fmt(bc['share_of_squared_gradient_norm'])} | "
         f"{bc['n_params_moved']:,} of {bc['n_params']:,} |"]
        + [gline(g) for g in grads])

    head = grads[0]
    hc = head["bond_gradient_contribution"]
    ht = head["target"]
    sec = sorted(head["by_section"].items(), key=lambda kv: -kv[1]["delta_sq"])
    sec_table = "\n".join(
        f"| `{k}` | {v['n']} | {fmt(v['delta_sq'])} | {fmt(v['sq'])} | {fmt(v['share_of_own'])} |"
        for k, v in sec if v["delta_sq"] > 0) or "| (none) | | | | |"

    sub = {
        "MASK_TABLE": mask_table,
        "GRAD_TABLE": grad_table,
        "SEC_TABLE": sec_table,
        "N_FIRED": str(mask["n_with_nonzero_bond_mask"]),
        "N_WALKED": str(mask["n_walked"]),
        "N_4G5J_FIRED": str(sum(1 for r in by_pdb.get("4g5j", []) if r["bond_mask_nnz"])),
        "N_4G5J": str(len(by_pdb.get("4g5j", []))),
        "N_4BYH_FIRED": str(sum(1 for r in by_pdb.get("4byh", []) if r["bond_mask_nnz"])),
        "N_4BYH": str(len(by_pdb.get("4byh", []))),
        "HEAD_TARGET": f"{ht['pdb_id']} {ht['datapoint']}",
        "HEAD_MASK": str(ht["bond_mask_nnz"]),
        "HEAD_TB": str(ht["token_bonds_nnz"]),
        "HEAD_BOND_LOSS": repr(head["arms"]["bond4"]["breakdown"].get("bond_loss")),
        "HEAD_DELTA": fmt(hc["squared_norm_of_difference"]),
        "HEAD_TOTAL": fmt(hc["squared_norm_with_bond"]),
        "HEAD_SHARE": fmt(hc["share_of_squared_gradient_norm"]),
        "HEAD_MOVED": f"{hc['n_params_moved']:,}",
        "HEAD_PARAMS": f"{hc['n_params']:,}",
        "HEAD_LOSS4": fmt(head["arms"]["bond4"]["loss"], 9),
        "HEAD_LOSS0": fmt(head["arms"]["bond0"]["loss"], 9),
        "HEAD_CROP": str(head["crop"]),
        "HEAD_STAGE": head["stage"],
        "HEAD_DATASET": head["dataset"],
        "HEAD_SEED": str(head["seed"]),
        "HEAD_DTYPE": head["dtype"],
        "BASE_MOVED": f"{bc['n_params_moved']:,}",
        "BASE_PARAMS": f"{bc['n_params']:,}",
        "MASK_PARTNERS": _partners(rows, ht["pdb_id"], ht["datapoint"]),
        "MASK_HEAD_TB": str(_headrow(rows, ht["pdb_id"], ht["datapoint"])["token_bonds_nnz"]),
        "MASK_HEAD_MASK": str(_headrow(rows, ht["pdb_id"], ht["datapoint"])["bond_mask_nnz"]),
        "MASK_CROP": str(mask["crop"]),
        "MASK_SEED": str(mask["seed"]),
        # counted from the directory, so the doc cannot claim a term with no record
        "DEMONSTRATED": str(sum(
            1 for f in sorted(a.evidence.glob("*.json"))
            if float(json.loads(f.read_text())["value"]) > 0)),
    }
    if mask2 is not None:
        r2 = mask2["rows"]
        b2: dict[str, list] = {}
        for r in r2:
            b2.setdefault(r["pdb_id"], []).append(r)
        sub.update({
            "MASK2_CROP": str(mask2["crop"]),
            "MASK2_SEED": str(mask2["seed"]),
            "MASK2_FIRED": str(mask2["n_with_nonzero_bond_mask"]),
            "MASK2_WALKED": str(mask2["n_walked"]),
            "MASK2_4G5J_FIRED": str(sum(1 for r in b2.get("4g5j", []) if r["bond_mask_nnz"])),
            "MASK2_4G5J": str(len(b2.get("4g5j", []))),
        })
    text = a.template.read_text()
    for k, v in sub.items():
        text = text.replace("{{" + k + "}}", v)
    left = [w for w in text.split("{{") if "}}" in w]
    if left:
        raise SystemExit(f"unsubstituted tokens: {[w.split('}}')[0] for w in left]}")
    a.out.write_text(text)
    print(f"wrote {a.out} ({len(text)}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
