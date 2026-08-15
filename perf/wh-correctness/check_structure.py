#!/usr/bin/env python3
"""Is this predicted structure chemically sane, and does it match what was asked for?

Written for the JapanFold correctness sweep: "it returned a file" is not a pass. The
failure this hunts is the plausible-looking wrong answer -- a fold that parses, carries
confidence numbers and silently dropped a chain, truncated a sequence, or folded a
sequence into spaghetti. Every check below is a detector for one of those, with a
threshold justified against ideal protein geometry rather than picked to look strict.

Usage:
    check_structure.py STRUCT.cif [--input target.yaml|target.fasta]
                       [--conf results.json] [--json out.json] [--quiet]

Exit 0 = every check passed, 1 = at least one FAIL, 2 = could not even parse.
WARN never changes the exit code; it marks a value worth a human's eye.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import gemmi
import numpy as np

# --- thresholds, each with the physical fact it comes from --------------------
CA_CA_BAND = (3.60, 4.10)  # +-0.2 A around the 3.805 A ideal trans peptide Ca-Ca
# Calibrated, not guessed: shipped tt-bio predictions measure 0.970 (opendde on
# 1ahw), 0.981 and 1.000 (boltz2) in-band, and the 1ahw crystal structure 0.986.
# So 0.98 as a failure bar rejects real output; a scrambled fold sits near 0.
CA_CA_FAIL_FRAC = 0.90   # below this the chain is not a protein backbone
CA_CA_WARN_FRAC = 0.98   # between the two: worth a human's eye, not a failure
CA_CA_BREAK = 5.0        # beyond this a *predicted* chain is physically broken
# Nucleic acids have no CA. Consecutive P-P spans ~5.9 A in A-form RNA and ~7.0 A
# in B-DNA, so one band covers both with room for local distortion.
PP_BAND = (5.0, 7.8)
PP_FAIL_FRAC = 0.85
PP_BREAK = 9.5
N_CA_BAND = (1.40, 1.52)   # ideal 1.458
CA_C_BAND = (1.47, 1.58)   # ideal 1.525
CLASH_DIST = 2.0         # heavy atoms >=2 residues apart never get this close
CLASH_MAX_FRAC = 0.001   # tolerate 0.1% of atoms in a marginal contact, not more
RG_COEF, RG_EXP = 2.2, 0.38  # Rg ~ 2.2 * N^0.38 A for a compact globular domain
RG_BAND = (0.55, 2.0)    # ratio to that estimate: <0.55 collapsed blob, >2 unfolded
PLDDT_STD_MIN = 0.01     # a constant-confidence output is the garbage signature
AA3 = None  # lazily filled from gemmi


def _one_letter(resname: str) -> str:
    info = gemmi.find_tabulated_residue(resname)
    if info is None:
        return "X"
    c = info.one_letter_code
    return c.upper() if c and c.isalpha() else "X"


# --- what was asked for -------------------------------------------------------
def parse_input(path: Path) -> dict:
    """Chains and ligands the input declares. Handles tt-bio YAML and plain FASTA.

    Deliberately independent of tt_bio.platform.limits: this script runs against
    outputs from the CLI as well as the API, and the platform tree is not always
    importable where the sweep runs.
    """
    text = path.read_text()
    want = {"chains": [], "ligands": [], "constraints": 0}
    if text.lstrip().startswith(">"):
        cur_id, cur, typ = None, [], "protein"
        for ln in text.splitlines():
            if ln.startswith(">"):
                if cur_id is not None:
                    want["chains"].append({"id": cur_id, "type": typ, "sequence": "".join(cur)})
                head = ln[1:].strip()
                parts = head.rsplit("|", 1)
                cur_id = (parts[0].split()[0] if parts[0].strip() else "A")
                typ = parts[1].strip().lower() if len(parts) == 2 else "protein"
                cur = []
            elif ln.strip():
                cur.append(re.sub(r"[^A-Za-z]", "", ln.strip()))
        if cur_id is not None:
            want["chains"].append({"id": cur_id, "type": typ, "sequence": "".join(cur)})
        return want

    import yaml
    data = yaml.safe_load(text) or {}
    for e in (data.get("sequences") or data.get("entities") or []):
        if not isinstance(e, dict):
            continue
        for key, body in e.items():
            if not isinstance(body, dict):
                continue
            k = str(key).lower()
            idv = body.get("id")
            ids = [str(x) for x in idv] if isinstance(idv, list) else [str(idv)]
            if k in ("protein", "dna", "rna"):
                seq = re.sub(r"[^A-Za-z]", "", str(body.get("sequence") or ""))
                # A declared modification legitimately changes the residue at that position,
                # so the sequence check has to know about it or it reports every correctly
                # applied modification as a mismatch. Positions are 1-based.
                mods = {int(m["position"]): str(m.get("ccd") or "")
                        for m in (body.get("modifications") or [])
                        if isinstance(m, dict) and m.get("position") is not None}
                for cid in ids:
                    want["chains"].append({"id": cid, "type": k, "sequence": seq,
                                           "modifications": mods})
            elif k == "ligand":
                for cid in ids:
                    want["ligands"].append({"id": cid, "smiles": body.get("smiles"),
                                            "ccd": body.get("ccd")})
    cons = data.get("constraints")
    want["constraints"] = len(cons) if isinstance(cons, list) else 0
    return want


# --- geometry -----------------------------------------------------------------
def chain_geometry(st: gemmi.Structure) -> tuple[list[dict], list[str], list[str]]:
    """Per-polymer-chain geometry, plus FAIL and WARN lines.

    Protein chains are measured on consecutive Ca; nucleic chains have no Ca, so
    they are measured on consecutive P (falling back to C1'). A chain with fewer
    than two anchors is a ligand or an ion and is left to the ligand count.
    """
    fails: list[str] = []
    warns: list[str] = []
    out: list[dict] = []
    for chain in st[0]:
        anchors, seq, ncas, cacs = [], [], [], []
        kind = "protein"
        for res in chain:
            a = res.find_atom("CA", "*")
            if a is None:
                a = res.find_atom("P", "*") or res.find_atom("C1'", "*")
                if a is not None:
                    kind = "nucleic"
            if a is None:
                continue
            anchors.append([a.pos.x, a.pos.y, a.pos.z])
            seq.append(_one_letter(res.name))
            if kind == "protein":
                n, c = res.find_atom("N", "*"), res.find_atom("C", "*")
                if n is not None:
                    ncas.append(n.pos.dist(a.pos))
                if c is not None:
                    cacs.append(a.pos.dist(c.pos))
        if len(anchors) < 2:
            continue
        band, fail_frac, brk = ((CA_CA_BAND, CA_CA_FAIL_FRAC, CA_CA_BREAK) if kind == "protein"
                                else (PP_BAND, PP_FAIL_FRAC, PP_BREAK))
        a = np.asarray(anchors)
        d = np.linalg.norm(a[1:] - a[:-1], axis=1)
        inband = float(((d >= band[0]) & (d <= band[1])).mean())
        breaks = int((d > brk).sum())
        rg = float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))
        rg_ref = RG_COEF * len(a) ** RG_EXP
        info = {"chain": chain.name, "kind": kind, "n_res": len(a), "sequence": "".join(seq),
                "step_median": round(float(np.median(d)), 3),
                "step_band": list(band),
                "in_band_frac": round(inband, 4),
                "breaks": breaks,
                "rg": round(rg, 2), "rg_expected": round(rg_ref, 2),
                "rg_ratio": round(rg / rg_ref, 3),
                "n_ca_bond_median": round(float(np.median(ncas)), 3) if ncas else None,
                "ca_c_bond_median": round(float(np.median(cacs)), 3) if cacs else None}
        out.append(info)
        tag = f"chain {chain.name} ({kind})"
        if inband < fail_frac:
            fails.append(f"{tag}: only {inband:.1%} of consecutive backbone steps in "
                         f"[{band[0]}, {band[1]}] A (floor {fail_frac:.0%})")
        elif inband < CA_CA_WARN_FRAC and kind == "protein":
            warns.append(f"{tag}: {inband:.1%} of Ca-Ca steps in band "
                         f"(shipped predictions measure 0.970-1.000)")
        if breaks:
            fails.append(f"{tag}: {breaks} backbone gap(s) > {brk} A -- chain is broken")
        # Rg is an empirical globular-protein relation; it says nothing about a
        # nucleic duplex, which is a rod, so only protein chains are judged on it.
        if kind == "protein" and not (RG_BAND[0] <= rg / rg_ref <= RG_BAND[1]):
            fails.append(f"{tag}: Rg {rg:.1f} A is {rg / rg_ref:.2f}x the {rg_ref:.1f} A "
                         f"expected for {len(a)} residues -- not a compact fold")
        for label, vals, bnd in (("N-Ca", ncas, N_CA_BAND), ("Ca-C", cacs, CA_C_BAND)):
            if not vals:
                continue
            med = float(np.median(vals))
            if not (bnd[0] <= med <= bnd[1]):
                fails.append(f"{tag}: median {label} bond {med:.3f} A outside {bnd}")
    return out, fails, warns


def clashes(st: gemmi.Structure) -> tuple[int, int, float]:
    """Heavy-atom pairs closer than CLASH_DIST, counting only residues >=2 apart
    (or on different chains) so real bonded and adjacent-residue contacts are not
    reported as clashes. Returns (n_clashes, n_heavy_atoms, worst_distance)."""
    # tt-bio's CIF writers emit a placeholder 1x1x1 A unit cell. Checked: the
    # neighbour search does not apply it, so no atom is paired with a periodic
    # image of itself (clearing the cell leaves the count identical, and a live
    # 1 A cell would produce ~6 images per atom, not the 57 pairs measured).
    model = st[0]
    ns = gemmi.NeighborSearch(st, 5.0).populate()
    n_clash, worst = 0, math.inf
    heavy = 0
    for ci, chain in enumerate(model):
        for res in chain:
            for atom in res:
                if atom.element == gemmi.Element("H"):
                    continue
                heavy += 1
                for m in ns.find_atoms(atom.pos, "\0", radius=CLASH_DIST):
                    cra = m.to_cra(model)
                    if cra.atom.element == gemmi.Element("H"):
                        continue
                    same_chain = cra.chain.name == chain.name
                    if same_chain and abs(cra.residue.seqid.num - res.seqid.num) < 2:
                        continue
                    dist = cra.atom.pos.dist(atom.pos)
                    if dist < CLASH_DIST and (cra.atom.serial != atom.serial):
                        n_clash += 1
                        worst = min(worst, dist)
    return n_clash // 2, heavy, (0.0 if worst is math.inf else round(worst, 3))


# --- confidence ---------------------------------------------------------------
def confidence(st: gemmi.Structure, conf_json: dict | None,
               required: bool = True) -> tuple[dict, list[str], list[str]]:
    """pLDDT from the results JSON when present, else from the B-factor column
    (every tt-bio writer puts per-atom pLDDT there). Checks range, and that it is
    not constant -- a flat confidence field is what a garbage fold produces."""
    fails, warns, info = [], [], {}
    vals = []
    if conf_json:
        for key in ("plddt", "pLDDT", "confidence_score", "mean_plddt"):
            v = conf_json.get(key)
            if isinstance(v, (int, float)):
                info["reported_" + key] = v
            elif isinstance(v, list) and v:
                vals = [float(x) for x in v]
    if not vals:
        vals = [a.b_iso for ch in st[0] for r in ch for a in r]
        info["source"] = "b_factor"
    else:
        info["source"] = "results_json"
    if not vals or not any(vals):
        msg = "no per-atom confidence in the structure or the results JSON"
        (fails if required else warns).append(msg)
        return info, fails, warns
    v = np.asarray(vals, dtype=float)
    if np.isnan(v).any() or np.isinf(v).any():
        fails.append(f"{int(np.isnan(v).sum() + np.isinf(v).sum())} non-finite confidence values")
        v = v[np.isfinite(v)]
    scale = 100.0 if v.max() > 1.5 else 1.0
    v = v / scale
    info.update({"n": int(v.size), "min": round(float(v.min()), 4),
                 "max": round(float(v.max()), 4), "mean": round(float(v.mean()), 4),
                 "std": round(float(v.std()), 4), "scale": scale})
    if v.min() < 0 or v.max() > 1.0001:
        fails.append(f"confidence outside [0,1] after scaling: [{v.min():.3f}, {v.max():.3f}]")
    if float(v.std()) < PLDDT_STD_MIN:
        (fails if required else warns).append(f"confidence is effectively constant (std {v.std():.4f} < {PLDDT_STD_MIN}) "
                     f"-- the signature of a garbage fold")
    if float(v.mean()) < 0.30:
        warns.append(f"mean confidence {v.mean():.3f} is very low")
    return info, fails, warns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("struct", type=Path)
    ap.add_argument("--input", type=Path, help="the YAML/FASTA that was submitted")
    ap.add_argument("--conf", type=Path, help="results JSON carrying confidence values")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--kind", default="predict", choices=("predict", "design"),
                    help="design outputs carry no pLDDT (they are scored by a separate "
                         "filter model), so confidence is reported but not required")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rep: dict = {"struct": str(a.struct), "checks": {}, "fail": [], "warn": []}
    try:
        st = gemmi.read_structure(str(a.struct))
        st.remove_alternative_conformations()
        st.setup_entities()
    except Exception as e:
        print(f"FAIL parse: {e}")
        return 2
    if len(st) == 0 or sum(1 for ch in st[0] for r in ch for _ in r) == 0:
        print("FAIL parse: no atoms")
        return 2

    coords = np.array([[at.pos.x, at.pos.y, at.pos.z] for ch in st[0] for r in ch for at in r])
    rep["checks"]["n_atoms"] = int(coords.shape[0])
    if not np.isfinite(coords).all():
        rep["fail"].append(f"{int((~np.isfinite(coords)).any(1).sum())} atoms have non-finite coordinates")
    extent = float(np.linalg.norm(coords.max(0) - coords.min(0))) if coords.size else 0.0
    rep["checks"]["bbox_diagonal"] = round(extent, 2)
    if extent < 5.0:
        rep["fail"].append(f"all atoms within a {extent:.1f} A box -- degenerate output")

    geo, gf, gw = chain_geometry(st)
    rep["checks"]["chains"] = geo
    rep["fail"] += gf
    rep["warn"] += gw

    n_clash, n_heavy, worst = clashes(st)
    rep["checks"]["clashes"] = {"n": n_clash, "heavy_atoms": n_heavy, "worst_dist": worst,
                                "threshold": CLASH_DIST}
    if n_heavy and n_clash > CLASH_MAX_FRAC * n_heavy:
        rep["fail"].append(f"{n_clash} heavy-atom clashes < {CLASH_DIST} A "
                           f"({n_clash / n_heavy:.2%} of atoms, worst {worst} A)")
    elif n_clash:
        rep["warn"].append(f"{n_clash} marginal contacts < {CLASH_DIST} A (worst {worst} A)")

    conf_json = json.loads(a.conf.read_text()) if a.conf and a.conf.exists() else None
    cinfo, cf, cw = confidence(st, conf_json, required=(a.kind == "predict"))
    rep["checks"]["confidence"] = cinfo
    rep["fail"] += cf
    rep["warn"] += cw

    if a.input:
        want = parse_input(a.input)
        rep["checks"]["expected"] = {"chains": len(want["chains"]),
                                     "ligands": len(want["ligands"])}
        got_poly = {g["chain"]: g for g in geo}
        # Ligands land as non-polymer residues; count them by residue, not chain.
        poly_names = set(got_poly)
        n_lig_res = sum(1 for ch in st[0] for r in ch
                        if ch.name not in poly_names or r.het_flag == "H")
        rep["checks"]["ligand_residues_seen"] = n_lig_res
        if len(got_poly) != len(want["chains"]):
            rep["fail"].append(f"expected {len(want['chains'])} polymer chain(s), "
                               f"found {len(got_poly)}: {sorted(got_poly)}")
        if want["ligands"] and n_lig_res < len(want["ligands"]):
            rep["fail"].append(f"expected {len(want['ligands'])} ligand(s), "
                               f"found {n_lig_res} non-polymer residue(s)")
        # Sequence identity per chain, matched by id when the ids survive.
        for want_ch in want["chains"]:
            got = got_poly.get(want_ch["id"])
            if got is None:
                continue
            exp, obs = want_ch["sequence"].upper(), got["sequence"].upper()
            if len(obs) != len(exp):
                rep["fail"].append(f"chain {want_ch['id']}: {len(obs)} residues out, "
                                   f"{len(exp)} in -- silently truncated or padded")
                continue
            mods = want_ch.get("modifications") or {}
            mism = sum(1 for i, (x, y) in enumerate(zip(exp, obs), 1)
                       if y not in (x, "X") and i not in mods)
            if mism:
                rep["fail"].append(f"chain {want_ch['id']}: {mism} residue(s) differ from "
                                   f"the submitted sequence")
            # A model that advertises `modifications` and quietly returns the unmodified
            # residue is the exact failure this sweep exists to hunt: the structure is
            # chemically sane, the confidence is real, and it answers a different question
            # than the one that was asked.
            if mods:
                by_name = {c.name: c for c in st[0]}
                res = list(by_name[want_ch["id"]]) if want_ch["id"] in by_name else []
                seen = {}
                for pos, ccd in sorted(mods.items()):
                    got_name = res[pos - 1].name.upper() if 0 < pos <= len(res) else "?"
                    seen[pos] = got_name
                    if ccd and got_name != ccd.upper():
                        rep["fail"].append(
                            f"chain {want_ch['id']}: modification {ccd} at position {pos} "
                            f"was not applied, the output carries {got_name}")
                rep["checks"]["modifications"] = {"requested": mods, "observed": seen}

    rep["verdict"] = "FAIL" if rep["fail"] else ("WARN" if rep["warn"] else "PASS")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=1))
    if not a.quiet:
        print(f"{rep['verdict']}  {a.struct}")
        for f in rep["fail"]:
            print("  FAIL " + f)
        for w in rep["warn"]:
            print("  WARN " + w)
        if rep["checks"].get("chains"):
            for g in rep["checks"]["chains"]:
                print(f"  chain {g['chain']} {g['kind']}: n={g['n_res']} "
                      f"step_med={g['step_median']} inband={g['in_band_frac']} "
                      f"rg_ratio={g['rg_ratio']}")
        c = rep["checks"]["confidence"]
        print(f"  confidence({c.get('source')}): mean={c.get('mean')} std={c.get('std')} "
              f"n={c.get('n')}")
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
