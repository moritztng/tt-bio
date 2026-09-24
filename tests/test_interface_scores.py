"""tt_bio.interface_scores against the script Adaptyv scored the Nipah competition with.

`fixtures/ipsae/nipah_ipsae_reference.py` is `ipsae.py` from
github.com/adaptyvbio/nipah_ipsae_pipeline @ 80e0d56, unmodified (MIT). Each case writes a
Boltz-layout mmCIF, PAE, pLDDT and confidence JSON, runs the reference on them as a subprocess,
and holds every column it prints to within half a unit of the last digit it prints.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tt_bio import interface_scores as isc

REF = Path(__file__).parent / "fixtures" / "ipsae" / "nipah_ipsae_reference.py"
FIELDS = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
          "label_seq_id", "auth_seq_id", "pdbx_PDB_ins_code", "label_asym_id", "Cartn_x",
          "Cartn_y", "Cartn_z", "occupancy", "label_entity_id", "auth_asym_id", "auth_comp_id",
          "B_iso_or_equiv", "pdbx_PDB_model_num"]
# (column in the reference's output, key in ours, decimals the reference prints)
COLUMNS = [("ipSAE", "ipsae", 6), ("ipSAE_d0chn", "ipsae_d0chn", 6), ("ipSAE_d0dom", "ipsae_d0dom", 6),
           ("ipTM_af", "iptm", 3), ("ipTM_d0chn", "iptm_d0chn", 6), ("pDockQ", "pdockq", 4),
           ("pDockQ2", "pdockq2", 4), ("LIS", "lis", 4)]
ASYM_COUNTS = [("n0res", "n0res"), ("n0chn", "n0chn"), ("n0dom", "n0dom"), ("nres1", "nres1"),
               ("nres2", "nres2"), ("dist1", "dist1"), ("dist2", "dist2")]
AA = ["ALA", "GLY", "LEU", "SER", "LYS", "GLU", "TRP", "GLY", "VAL", "ASP"]


def _complex(rng, lengths, ligand_atoms=0, nucleic=None, grid=False):
    """Chains laid side by side along x, close enough to touch, as Boltz mmCIF atom rows."""
    rows, n_tok = [], 0
    for ci, n in enumerate(lengths):
        cid = chr(65 + ci)
        na = nucleic == ci
        for r in range(n):
            res = ("DA", "DC", "DG", "DT")[r % 4] if na else AA[(r * 7 + ci) % len(AA)]
            base = np.array([ci * 7.0 + rng.normal(0, 1.5), r * 3.8 - n * 1.9, rng.normal(0, 3.0)])
            atoms = (["C1'", "C3'", "P"] if na else ["N", "CA", "C", "O"] + ([] if res == "GLY" else ["CB"]))
            for a in atoms:
                xyz = base + rng.normal(0, 0.8, 3)
                rows.append((a, res, r + 1, cid, np.round(xyz) if grid else xyz, ci + 1))
            n_tok += 1
    if ligand_atoms:
        cid = chr(65 + len(lengths))
        for k in range(ligand_atoms):
            rows.append((f"C{k + 1}", "LIG", ".", cid, np.array([3.5, 0.0, 0.0]) + rng.normal(0, 1.2, 3),
                         len(lengths) + 1))
        n_tok += ligand_atoms
    lines = ["data_model", "loop_"] + [f"_atom_site.{f}" for f in FIELDS]
    for i, (a, res, seq, cid, xyz, ent) in enumerate(rows):
        grp = "HETATM" if seq == "." else "ATOM"
        lines.append(f"{grp} {i + 1} {a[0]} {a} . {res} {seq} {seq if seq != '.' else 1} ? {cid} "
                     f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f} 1 {ent} {cid} {res} 50.0 1")
    return "\n".join(lines) + "\n", n_tok


def _case(tmp_path, seed, lengths, ligand_atoms=0, nucleic=None, iptm=True, grid=False):
    """`grid` puts coordinates on integers and PAE on half-Angstrom steps, so pairs land exactly
    on the 8 A contact and the PAE cutoffs and the `<` / `<=` choices are exercised."""
    rng = np.random.default_rng(seed)
    cif, n = _complex(rng, lengths, ligand_atoms, nucleic, grid)
    d = tmp_path / f"case{seed}"
    d.mkdir()
    (d / "m_model_0.cif").write_text(cif)
    # A PAE that is low inside chains and spans the cutoff between them, float32 as Boltz writes it.
    chain_of = np.repeat(np.arange(len(lengths) + (1 if ligand_atoms else 0)),
                         list(lengths) + ([ligand_atoms] if ligand_atoms else []))
    same = chain_of[:, None] == chain_of[None, :]
    pae = np.where(same, rng.uniform(0.5, 6, (n, n)), rng.gamma(3.0, 4.5, (n, n)))
    pae = (np.round(pae * 2) / 2 if grid else pae).astype(np.float32)
    np.savez(d / "pae_m_model_0.npz", pae=pae)
    np.savez(d / "plddt_m_model_0.npz", plddt=rng.uniform(0.3, 0.98, n).astype(np.float32))
    k = len(lengths) + (1 if ligand_atoms else 0)
    if iptm:
        m = rng.uniform(0.1, 0.9, (k, k))
        (d / "confidence_m_model_0.json").write_text(json.dumps(
            {"pair_chains_iptm": {str(i): {str(j): float(m[i, j]) for j in range(k)} for i in range(k)}}))
    return d


def _reference(d, pae_cutoff, dist_cutoff):
    subprocess.run([sys.executable, str(REF), str(d / "pae_m_model_0.npz"), str(d / "m_model_0.cif"),
                    str(pae_cutoff), str(dist_cutoff)], check=True, capture_output=True, cwd=d)
    lines = [ln for ln in (d / f"m_model_0_{int(pae_cutoff):02d}_{int(dist_cutoff):02d}.txt")
             .read_text().splitlines() if ln.strip()]
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","))) for ln in lines[1:]]


@pytest.mark.parametrize("seed,lengths,ligand,nucleic,cut,grid", [
    (1, (40, 30), 0, None, (15.0, 15.0), False),        # the Nipah pipeline's call
    (2, (55, 23), 0, None, (10.0, 10.0), False),        # Dunbrack's recommended cutoffs
    (3, (25, 30, 18), 6, None, (15.0, 15.0), False),     # three chains and a ligand the mask must drop
    (4, (35, 12), 0, 1, (15.0, 15.0), False),            # a nucleic-acid partner, d0 floor of 2
    (5, (8, 9), 0, None, (4.0, 15.0), False),
    (6, (40, 30), 0, None, (15.0, 15.0), True),  # ties on every cutoff
    (9, (30, 26), 0, None, (10.0, 12.0), True),            # below the 27-residue d0 floor, sparse PAE hits
])
def test_matches_nipah_reference(tmp_path, seed, lengths, ligand, nucleic, cut, grid):
    d = _case(tmp_path, seed, lengths, ligand, nucleic, grid=grid)
    ref = _reference(d, *cut)
    ours = isc.score_files(d / "m_model_0.cif", d / "pae_m_model_0.npz", d / "plddt_m_model_0.npz",
                           d / "confidence_m_model_0.json", *cut)
    checked = 0
    for row in ref:
        a, b = row["Chn1"], row["Chn2"]
        if row["Type"] == "asym":
            got = ours["directions"][f"{a}->{b}"]
        else:
            got = ours["pairs"]["-".join(sorted((a, b)))]
        for col, key, dec in COLUMNS:
            assert abs(float(row[col]) - got[key]) <= 0.5 * 10 ** -dec + 1e-9, (row["Type"], a, b, col,
                                                                              row[col], got[key])
            checked += 1
        if row["Type"] == "asym":
            for col, key in ASYM_COUNTS:
                assert int(row[col]) == got[key], (a, b, col, row[col], got[key])
    assert checked >= len(COLUMNS) * 3


def test_ipsae_min_and_interface_pae(tmp_path):
    d = _case(tmp_path, 7, (30, 20))
    s = isc.score_files(d / "m_model_0.cif", d / "pae_m_model_0.npz", d / "plddt_m_model_0.npz")
    p, ab, ba = s["pairs"]["A-B"], s["directions"]["A->B"], s["directions"]["B->A"]
    assert p["ipsae_min"] == min(ab["ipsae"], ba["ipsae"]) and p["ipsae"] == max(ab["ipsae"], ba["ipsae"])
    pae = np.load(d / "pae_m_model_0.npz")["pae"].astype(float)
    assert p["interface_pae"] == pytest.approx(np.concatenate([pae[:30, 30:].ravel(), pae[30:, :30].ravel()]).mean())
    assert p["iptm"] is None and s["version"] == isc.SCORES_VERSION


def test_missing_plddt_floors_pdockq_like_the_reference(tmp_path):
    d = _case(tmp_path, 8, (30, 30))
    s = isc.score_files(d / "m_model_0.cif", d / "pae_m_model_0.npz")
    assert s["pairs"]["A-B"]["pdockq2"] == pytest.approx(1.31 / (1 + np.exp(0.075 * 84.733)) + 0.005)


def test_boltzgen_design_head_differs_only_below_the_reference_d0_floor():
    """BoltzGen's `compute_ipsae_score` is the same per-residue ipSAE with a d0 floor of 19
    residues instead of 27. Holds that it agrees to float precision when the best residue has
    27+ partners under the cutoff, and that the gap below that stays under 0.01."""
    torch = pytest.importorskip("torch")
    from tt_bio.boltzgen.model.layers.confidence_utils import compute_ipsae_score
    rng = np.random.default_rng(0)
    for _ in range(60):
        n1, n2 = int(rng.integers(40, 150)), int(rng.integers(60, 250))
        n = n1 + n2
        chains = np.array(["B"] * n1 + ["A"] * n2)
        same = chains[:, None] == chains[None, :]
        frac = rng.uniform(0.0, 0.6)
        inter = np.where(rng.random((n, n)) < frac, rng.uniform(1, 12, (n, n)), rng.uniform(12, 31, (n, n)))
        pae = np.where(same, rng.uniform(0.5, 5, (n, n)), inter)
        ref = isc.score(pae, np.full(n, 80.0), chains, rng.normal(0, 20, (n, 3)))["directions"]
        P = torch.tensor(pae)[None]
        one = torch.ones(1, n, dtype=P.dtype)
        b = torch.tensor(chains == "B", dtype=P.dtype)[None]
        for src, tgt, key in ((b, 1 - b, "B->A"), (1 - b, b, "A->B")):
            got = compute_ipsae_score(src, tgt, P, one, one).item()
            gap = abs(got - ref[key]["ipsae"])
            assert gap < (1e-6 if ref[key]["n0res"] >= 27 else 1e-2), (key, ref[key]["n0res"], gap)
