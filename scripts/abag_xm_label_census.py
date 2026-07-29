#!/usr/bin/env python3
"""AbAg-XM label census (closeout spec 1.1): every null-label fold x column,
with a verified cause.

The dataset has 164 targets; 161 are scorable. The 3 exclusions (9ly2, 9ly3,
9lz2) carry null `dockq`: they are anti-phosphoepitope antibodies whose
native interface is carried by SEP (phosphoserine) residues that DockQ's
loader discards, so no scorable interface atoms exist and the null is the
correct record (verified: antigen chain present and resolved, SEP residues
in the antigen chain). Two more null classes exist (`interface_lddt`,
`cdr_h3_rmsd`) and each fold gets a cause assigned here from the labels JSON
error fields plus a native-side sequence check:

  * error says the yaml chain was "not found in model/native by sequence":
    if a native chain's sanitized polymer sequence DOES match (exact,
    containment, or >=0.95 identity), the null is the exact-match
    chain-resolution bug class (recoverable by prefix-match); otherwise the
    native chain is genuinely (substantially) unresolved.

Counts are asserted against the merged ranker table: 9 dockq-null folds (450
samples), 13 interface_lddt-null folds (603 samples: the 3 dockq-null targets
x3 gens, 9mz8 x3 gens, 9mnu/boltz2 ranks 6/18/30), 15 cdr_h3_rmsd-null folds
(750 samples).

    python3 scripts/abag_xm_label_census.py \
        --csv ~/abag_xm/tier_a/ranker_scores.csv --out_dir docs

Outputs docs/abag-xm-label-census.{md,csv}.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

GEN_PREFIX = {"opendde-abag": "opendde_abag", "protenix-v2": "protenix_v2",
              "boltz2": "boltz2"}
COLS = ("dockq", "interface_lddt", "cdr_h3_rmsd")
ROLE = {"dockq": "antigen", "interface_lddt": "antigen", "cdr_h3_rmsd": "H"}


def _sanitize(seq):
    return "".join(c if c in "ACDEFGHIKLMNPQRSTVWY" else "X"
                   for c in (seq or "").replace("-", ""))


def _native_match(gt_cif, yaml_path, role):
    """Match the yaml chain for `role` against native polymer sequences.

    Returns (matched, resolved_ca, yaml_len) where resolved_ca counts CA atoms
    in the best-matching native chain -- the polymer SEQUENCE exists in the
    mmCIF even where every residue is unresolved, so only the atom count
    distinguishes "substantially unresolved" from "present".
    """
    import gemmi
    import yaml
    ys = {e["protein"]["id"]: e["protein"]["sequence"]
          for e in yaml.safe_load(Path(yaml_path).read_text())["sequences"]}
    yseq = _sanitize(ys.get(role if role != "antigen" else "A", ""))
    matched = False
    resolved_ca = 0
    try:
        st = gemmi.read_structure(str(gt_cif))
        for m in st:
            for ch in m:
                try:
                    s = _sanitize(ch.get_polymer().make_one_letter_sequence())
                except Exception:
                    continue
                if not yseq or not s:
                    continue
                # short antigens (peptides): a single mismatch is >5%, so
                # allow <=2 mismatches under 30 aa
                bar = 0.95 if len(yseq) >= 30 else 1.0 - 2.0 / max(len(yseq), 1)
                if (s == yseq or s in yseq or yseq in s or
                        (len(s) == len(yseq) and
                         sum(a == b for a, b in zip(s, yseq)) / len(s) >= bar)):
                    matched = True
                    n_ca = sum(1 for r in ch for at in r if at.name == "CA")
                    resolved_ca = max(resolved_ca, n_ca)
            break
    except Exception:
        pass
    return matched, resolved_ca, len(yseq)


def _has_sep(gt_cif):
    """Native carries phosphoserine (SEP) residues (anti-phosphoepitope class)."""
    import gemmi
    try:
        st = gemmi.read_structure(str(gt_cif))
        return any(r.name == "SEP" for m in st for ch in m for r in ch)
    except Exception:
        return False


def _evidence(rec):
    """Pull a short cause string out of a per-sample label record."""
    if not isinstance(rec, dict):
        return "missing record"
    for k in ("_error", "error", "status"):
        if rec.get(k):
            return str(rec[k])[:200]
    if rec.get("n_interface_residues") == 0:
        return "n_interface_residues: 0 (model pose docked away from the " \
               "antigen -- correct null)"
    if rec.get("_raw"):
        return "raw: " + str(rec["_raw"])[:180]
    if not rec:
        return "empty record"
    scalars = {k: v for k, v in rec.items()
               if not isinstance(v, (list, dict)) and v is None}
    return "null fields: " + ",".join(sorted(scalars)) if scalars else \
        "keys: " + ",".join(sorted(rec)[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path.home() / "abag_xm/tier_a/ranker_scores.csv"))
    ap.add_argument("--labels_dir", default=str(Path.home() / "abag_xm/tier_a/labels"))
    ap.add_argument("--gt_dir", default=str(Path.home() / "abag_xm/ground_truth"))
    ap.add_argument("--yaml_dir", default=str(Path(__file__).resolve().parent.parent
                                            / "examples/abag_xm"))
    ap.add_argument("--out_dir", default="docs")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    rows = []
    for (target, gen), g in df.groupby(["target", "gen"]):
        for col in COLS:
            n_null = int(g[col].isna().sum())
            if not n_null:
                continue
            lp = Path(a.labels_dir) / f"{GEN_PREFIX[gen]}_{target}.json"
            ev = ""
            if lp.exists():
                d = json.loads(lp.read_text())
                key = {"dockq": "dockq", "interface_lddt": "interface_lddt",
                       "cdr_h3_rmsd": "cdr_rmsd"}[col]
                for s in d["samples"]:
                    rec = s.get(key) or {}
                    val = rec.get(col) if key != "cdr_rmsd" else None
                    if val is None:
                        ev = _evidence(rec)
                        if ev:
                            break
            matched, n_ca, ylen = _native_match(
                Path(a.gt_dir) / f"{target}.cif",
                Path(a.yaml_dir) / f"{target}.yaml", ROLE[col])
            gt = Path(a.gt_dir) / f"{target}.cif"
            if col == "dockq" and _has_sep(gt):
                cause = ("anti-phosphoepitope: the native interface is carried by "
                         "SEP (phosphoserine) residues that DockQ's loader "
                         "discards -- no scorable interface atoms")
            elif "n_interface_residues: 0" in ev:
                cause = ("model pose docked away from the antigen "
                         "(n_interface_residues: 0) -- correct null")
            elif "bitscore" in ev or "hmmer" in ev.lower():
                cause = ("harness: ANARCI species-limit warning on stdout broke "
                         "JSON parsing (record truncated); recoverable by "
                         "re-running the CDR script")
            elif col == "cdr_h3_rmsd" and "cdrs" in ev:
                cause = ("no heavy-chain CDR numbering recovered (H1 null, "
                         "H2/H3 absent; light-chain CDRs scored) -- consistent "
                         "with the native heavy chain unresolved at CDR-H3")
            elif (not matched) or n_ca < max(min(20, ylen), 0.3 * ylen):
                detail = (f"no matching polymer" if not matched else
                          f"only {n_ca}/{ylen} residues with atoms")
                cause = (f"native {ROLE[col]} chain substantially unresolved "
                         f"({detail}) -- no scorable interface" if col == "dockq"
                         else f"native {ROLE[col]} chain substantially unresolved "
                              f"({detail})")
            elif "not found" in ev:
                cause = "recoverable: exact-match chain-resolution bug " \
                        f"(native chain present, {n_ca}/{ylen} residues with atoms)"
            else:
                cause = "native chain resolved; label pipeline null (see evidence)"
            rows.append({"target": target, "gen": gen, "column": col,
                         "n_null": n_null, "n_samples": len(g), "cause": cause,
                         "evidence": ev})
    cen = pd.DataFrame(rows).sort_values(["column", "target", "gen"])
    out_csv = Path(a.out_dir) / "abag-xm-label-census.csv"
    cen.to_csv(out_csv, index=False)

    # ---- accept-criteria assertions --------------------------------------
    n_dockq = cen[cen.column == "dockq"]
    n_lddt = cen[cen.column == "interface_lddt"]
    n_cdr = cen[cen.column == "cdr_h3_rmsd"]
    fails = []
    if len(n_dockq) != 9 or n_dockq.n_null.sum() != 450:
        fails.append(f"dockq null folds {len(n_dockq)} != 9 "
                     f"({n_dockq.n_null.sum()} samples != 450)")
    if len(n_lddt) != 13 or n_lddt.n_null.sum() != 603:
        fails.append(f"interface_lddt null folds {len(n_lddt)} != 13 "
                     f"({n_lddt.n_null.sum()} samples != 603)")
    if len(n_cdr) != 15 or n_cdr.n_null.sum() != 750:
        fails.append(f"cdr_h3_rmsd null folds {len(n_cdr)} != 15 "
                     f"({n_cdr.n_null.sum()} samples != 750)")
    if cen.cause.eq("").any():
        fails.append("census rows without a cause")

    md = ["# AbAg-XM label census", "",
          "164 targets; 161 scorable (the 3 anti-phosphoepitope targets 9ly2, "
          "9ly3, 9lz2 have no scorable native interface -- their contacts are "
          "carried by phosphoserine residues that DockQ's loader discards). "
          "Every success-rate table uses denominator 161. This census lists "
          "every null-label fold x column with its verified cause.", ""]
    for col in COLS:
        sub = cen[cen.column == col]
        md.append(f"## `{col}` -- {len(sub)} folds, {int(sub.n_null.sum())} samples")
        md.append("")
        md.append("| target | gen | null/50 | cause |")
        md.append("|---|---|---|---|")
        for _, r in sub.iterrows():
            md.append(f"| {r.target} | {r.gen} | {r.n_null}/{r.n_samples} | "
                      f"{r.cause} |")
        md.append("")
    out_md = Path(a.out_dir) / "abag-xm-label-census.md"
    out_md.write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(f"... wrote {out_md} and {out_csv}")
    if fails:
        for f in fails:
            print("FAIL:", f)
        sys.exit(1)
    print("census accept criteria pass: 9/13/15 folds, 450/603/750 samples, "
          "every row has a cause")


if __name__ == "__main__":
    main()
