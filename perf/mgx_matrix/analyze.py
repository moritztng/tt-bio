"""Turn perf/mgx_matrix/out/<model>/ into the measured model x input table.

For every (model, input) this records what happened -- folded, refused (with the message), or
failed (with the last error line) -- and, for a fold, whether the feature the input carries is
actually in the structure. A status=ok fold that dropped the ligand is the failure this campaign
exists to find, so "folded" alone is never the verdict for a feature cell:

  ligand / rna / dna  the chain is in the output with atoms, not clashing into the protein
  cyclic              the N(1)-C(last) peptide bond is closed (< 2.0 A)
  modification        residue 7 is written as TPO and carries its phosphorus
  bond                the two named atoms sit at covalent distance (< 2.5 A)
  pocket              the ligand touches residue 44 or 70 (heavy-atom distance <= 6 A, the
                      constraint's own default)
  contact             A44 and B6 within 8 A (the contact default)
  affinity            results.json carries an affinity value
  template            CA RMSD to 1a8q.cif with and without the template (a template that changes
                      nothing is a dropped template)

Usage: analyze.py <out_root> <inputs_dir>  -> JSON on stdout
"""
import json
import re
import sys
from pathlib import Path

import gemmi
import numpy as np

OUT, INPUTS = Path(sys.argv[1]), Path(sys.argv[2])
ROOT = INPUTS.parents[2]
# Input chain ids in reader order, per input file, from the static pass. The checks name chains
# by the id the INPUT gave them and map to the output by order, because protenix and opendde
# write chains as A, B, C... whatever the input called them (recorded separately as
# "chain_ids_kept").
STATIC_ALL = json.loads((INPUTS.parent / "static_pass.json").read_text())
STATIC = STATIC_ALL["boltz2"]
# Heavy-atom distances. Below CLASH two atoms overlap; a bonded pair must also sit above it, so a
# bond constraint that pulls SG onto C1 at 0.9 A is reported as an overlap, not as honoured.
CLASH, BOND_MAX = 1.0, 2.5


def last_error(log: Path) -> str:
    lines = [l.strip() for l in log.read_text(errors="replace").splitlines() if l.strip()]
    for l in reversed(lines):
        if re.search(r"(Error|error:|Exception|cannot honour|refus|Traceback|FATAL|Aborted)", l) \
                and not l.startswith("EXIT="):
            return l[:400]
    return (lines[-2] if len(lines) > 1 else "")[:400]


def refusal_text(log: Path) -> str:
    t = log.read_text(errors="replace")
    m = re.search(r"Error: (.*?)(?:\nEXIT=|\Z)", t, re.S)
    return (m.group(1).strip() if m else last_error(log))[:1200]


def structure_for(model_dir: Path, stem: str):
    """The best-ranked structure and the results.json row for one input of a batch run."""
    for d in model_dir.glob("*_results_*"):
        rows = []
        rj = d / "results.json"
        if rj.is_file():
            try:
                rows = json.loads(rj.read_text())
            except json.JSONDecodeError:
                rows = []
        row = next((r for r in rows if isinstance(r, dict) and r.get("id") == stem), None)
        cifs = sorted(p for p in (d / "structures").glob(f"{stem}.*") if p.suffix in (".cif", ".pdb"))
        if cifs or row:
            return (cifs[0] if cifs else None), row
    return None, None


def atoms(st):
    out = []
    for ch in st[0]:
        for i, r in enumerate(ch):
            for a in r:
                out.append((ch.name, i + 1, r.name, a.name, a.element.name,
                            np.array(a.pos.tolist())))
    return out


def chain_summary(st):
    return {ch.name: {"residues": len(ch), "atoms": sum(len(r) for r in ch),
                      "first": ch[0].name if len(ch) else None} for ch in st[0]}


def min_dist(A, B):
    if not A or not B:
        return None
    a = np.stack([x[5] for x in A]); b = np.stack([x[5] for x in B])
    return float(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min())


def ca_rmsd_to(st, ref_path):
    ref = gemmi.read_structure(str(ref_path))
    ref_ca = [r["CA"][0].pos for r in ref[0]["A"] if r.find_atom("CA", "*")] \
        if ref[0].find_chain("A") else []
    ca = [r["CA"][0].pos for r in st[0][0] if r.find_atom("CA", "*")]
    n = min(len(ca), len(ref_ca))
    if n < 10:
        return None
    # 1a8q.cif is the template structure and the input is its own sequence; align by index.
    P = np.array([p.tolist() for p in ca[:n]]); Q = np.array([p.tolist() for p in ref_ca[:n]])
    P -= P.mean(0); Q -= Q.mean(0)
    U, _S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))


def feature_check(name, stem, st, res):
    at = atoms(st)
    want = [cid for cid, _mt, _n in STATIC.get(name, {}).get("chains", [])]
    have = [ch.name for ch in st[0]]
    alias = dict(zip(want, have)) if len(want) == len(have) else {}
    by_chain = lambda c: [x for x in at if x[0] == alias.get(c, c)]
    heavy = lambda xs: [x for x in xs if x[4] != "H"]
    chains = chain_summary(st)
    c = {"chains": chains, "chain_ids_kept": want == have if want else None}
    if stem in ("ligand_ccd", "ligand_smiles", "affinity", "fasta_ligand", "pocket"):
        lig = heavy(by_chain("L"))
        c["ligand_atoms"] = len(lig)
        c["ligand_protein_min_dist"] = min_dist(lig, heavy(by_chain("A")))
        c["present"] = bool(lig)
        c["clash"] = (c["ligand_protein_min_dist"] or 0) <= CLASH
        c["ok"] = c["present"] and not c["clash"]
    if stem in ("rna", "dna", "rna_only"):
        cid = "D" if stem == "dna" else "R"
        na = heavy(by_chain(cid))
        c["na_residues"] = chains.get(alias.get(cid, cid), {}).get("residues", 0)
        c["na_protein_min_dist"] = min_dist(na, heavy(by_chain("A"))) if stem != "rna_only" else None
        c["present"] = c["na_residues"] > 0
        c["clash"] = stem != "rna_only" and (c["na_protein_min_dist"] or 0) <= CLASH
        c["ok"] = c["present"] and not c["clash"]
    if stem == "cyclic":
        A = by_chain("A")
        n = [x for x in A if x[1] == 1 and x[3] == "N"]
        last = max(x[1] for x in A)
        cc = [x for x in A if x[1] == last and x[3] == "C"]
        c["n_c_dist"] = min_dist(n, cc)
        c["ok"] = c["n_c_dist"] is not None and c["n_c_dist"] < 2.0
    if stem == "modification":
        r7 = [x for x in by_chain("A") if x[1] == 7]
        c["res7_name"] = r7[0][2] if r7 else None
        c["res7_has_P"] = any(x[4] == "P" for x in r7)
        c["ok"] = c["res7_name"] == "TPO" and c["res7_has_P"]
    if stem.startswith("bond_"):
        spec = {"bond_protein_cys": (("A", 2, "SG"), ("B", 4, "SG")),
                "bond_protein_shipped_example": (("A", 7, "SG"), ("B", 5, "SG")),
                "bond_ligand": (("A", 2, "SG"), ("B", 1, "C1"))}[stem]
        pick = lambda ch, ri, an: [x for x in at if x[0] == alias.get(ch, ch) and x[1] == ri
                                   and x[3] == an]
        a1, a2 = pick(*spec[0]), pick(*spec[1])
        c["atom1_found"], c["atom2_found"] = bool(a1), bool(a2)
        c["bond_dist"] = min_dist(a1, a2)
        c["present"] = c["bond_dist"] is not None and c["bond_dist"] < BOND_MAX
        c["clash"] = c["bond_dist"] is not None and c["bond_dist"] <= CLASH
        c["ok"] = c["present"] and not c["clash"]
    if stem == "pocket":
        lig = heavy(by_chain("L"))
        pk = heavy([x for x in by_chain("A") if x[1] in (44, 70)])
        c["pocket_min_dist"] = min_dist(lig, pk)
        c["ok"] = c["pocket_min_dist"] is not None and c["pocket_min_dist"] <= 6.0
    if stem == "contact":
        a = heavy([x for x in by_chain("A") if x[1] == 44])
        b = heavy([x for x in by_chain("B") if x[1] == 6])
        c["contact_min_dist"] = min_dist(a, b)
        c["ok"] = c["contact_min_dist"] is not None and c["contact_min_dist"] <= 8.0
    if stem == "affinity":
        c["affinity_keys"] = sorted(k for k in (res or {}) if k.startswith("affinity_"))
        c["affinity_in_results"] = bool(c["affinity_keys"])
        c["ok"] = c.get("ok", True) and c["affinity_in_results"]
    if stem.startswith("template_"):
        c["ca_rmsd_to_1a8q"] = ca_rmsd_to(st, ROOT / "examples/of3_upstream/template_structures/1a8q.cif")
    if stem == "homomer":
        c["ok"] = len(chains) == 2
    if stem == "heteromer3":
        c["ok"] = len(chains) == 3
    b = [a.b_iso for ch in st[0] for r in ch for a in r]
    c["mean_bfactor"] = float(np.mean(b)) if b else None
    return c


table = {}
for model_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
    model = model_dir.name
    run_log = model_dir / "run.log"
    run_txt = run_log.read_text(errors="replace") if run_log.is_file() else ""
    run_tail = run_txt[-4000:]
    finished = "EXIT=" in run_txt
    static = STATIC_ALL.get(model, {})
    row = {}
    for inp in sorted(INPUTS.iterdir()):
        stem = inp.stem
        ref_log = model_dir / f"refused_{inp.name}.log"
        if ref_log.is_file():
            txt = ref_log.read_text(errors="replace")
            code = re.search(r"EXIT=(\d+)", txt)
            row[inp.name] = {"outcome": "refused", "exit": int(code.group(1)) if code else None,
                             "message": refusal_text(ref_log)}
            continue
        cif, res = structure_for(model_dir, stem)
        if cif is None:
            m = re.search(rf"{re.escape(stem)}[^\n]*\n(?:[^\n]*\n){{0,6}}?[^\n]*(Error|error|Exception|failed)[^\n]*",
                          run_tail)
            if res and res.get("status") == "failed":
                outcome = "failed"
            elif not run_log.is_file():
                outcome = "not_run"
            else:
                outcome = "failed" if finished else "in_flight"
            row[inp.name] = {"outcome": outcome,
                             "detail": (m.group(0)[-400:] if m else last_error(run_log)
                                        if run_log.is_file() else None),
                             "results": res}
            continue
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        chk = feature_check(inp.name, stem, st, res)
        row[inp.name] = {"outcome": "folded", "structure": str(cif.relative_to(OUT)), "check": chk}
        notes = static.get(inp.name, {}).get("notes")
        if notes:
            # The static pass printed a "Note: ... ignores" line: a dropped feature that was
            # announced. Record it, so a noted drop is scored as warned and not as broken.
            row[inp.name]["warned"] = notes
    for stem in ("template_npz", "template_cif"):
        on, off = row.get(f"{stem}.yaml", {}), row.get("template_off.yaml", {})
        r_on = on.get("check", {}).get("ca_rmsd_to_1a8q")
        r_off = off.get("check", {}).get("ca_rmsd_to_1a8q")
        if r_on is not None and r_off is not None:
            on["check"]["rmsd_delta_vs_off"] = r_on - r_off
            on["check"]["ok"] = abs(r_on - r_off) > 0.05
        elif r_on is not None:
            # A template is only scored against the same model without it; until that control
            # has folded, the cell is open, not ok.
            on["outcome"] = "awaiting_control"
    table[model] = row
json.dump(table, sys.stdout, indent=1, default=str)
