"""Compute the literature's best closed-form Ab-Ag trust metrics (AntiConf, pDockQ2,
ipSAE) from a tt-bio fold's winner CIF + --write_pae dump, so they can be compared
head-to-head against Protenix-v2 native ipTM / self-consistency (Arm 2 of the
flagship-abag-trust-validation task).

Formulas verified against primary sources this pass (see
~/.coworker/state/flagship-abag-trust-validation.md for citations):
  - pDockQ2 (Zhu et al. 2023, Bioinformatics 39(7):btad424): sigmoid of
    mean(1/(1+(PAE_int/d0)^2)) * mean(pLDDT_int) over CA-CA<8A interface residues.
    Fitted constants (L, x0, k, b) taken verbatim from the af-analysis reference
    implementation (github.com/samuelmurail/af_analysis, pip af-analysis==0.2.1,
    analysis.py:compute_pdockQ2) -- an independent, published, pip-installable
    implementation, not re-derived from a paraphrase.
  - ipSAE (Dunbrack et al., bioRxiv 2025.02.10.637595): per-residue pTM-like score
    1/(1+(PAE/d0res)^2) averaged over interchain residue pairs with PAE<pae_cutoff
    (default 10A), d0 from calc_d0(L) using the AF/TM-score d0 formula.
  - AntiConf (Briefings in Bioinformatics 27(2):bbag137, 2026): 0.7*pTM + 0.3*pDockQ2.

Assumes CIF chain order == yaml chain declaration order (antigen first, then
antibody heavy [, light]) -- true for every yaml this campaign uses, since
build_complex_features/_read_bio_chains preserve declaration order and the CIF
writer emits chains in that same order under renamed IDs (A, B, C, ...). Sanity-
checked per call: raises if a chain's residue count doesn't match len(sequence).
"""
import json, math, re
import numpy as np
import gemmi

# --- pDockQ2 constants, verbatim from af-analysis 0.2.1's compute_pdockQ2 ---
PDOCKQ2_L = 1.31034849e00
PDOCKQ2_X0 = 8.47326239e01
PDOCKQ2_K = 7.47157696e-02
PDOCKQ2_B = 5.01886443e-03
PDOCKQ2_D0 = 10.0
PDOCKQ2_CUTOFF_A = 8.0  # CA-CA distance cutoff for interface residues

IPSAE_PAE_CUTOFF = 10.0  # Angstrom, Dunbrack et al.'s recommended default


def calc_d0(L, min_value=1.0):
    """AF/TM-score d0, as used by ipSAE (ipsae.py:calc_d0)."""
    if L > 27:
        return max(min_value, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)
    return min_value


def parse_yaml_chains(yaml_path):
    """Same minimal parse as abag_decoy_build.py: returns [(id, sequence), ...] in
    file declaration order."""
    chains = []
    cur_id = None
    with open(yaml_path) as f:
        for line in f:
            m = re.match(r"\s*id:\s*(\S+)", line)
            if m:
                cur_id = m.group(1)
                continue
            m = re.match(r"\s*sequence:\s*(\S+)", line)
            if m and cur_id:
                chains.append((cur_id, m.group(1)))
                cur_id = None
    return chains


def load_cif_chains(cif_path):
    """Returns list of dicts per chain (file order): {ca_xyz: (N,3), plddt: (N,), n: int}."""
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    model = st[0]
    out = []
    for chain in model:
        xyz, plddt = [], []
        for res in chain:
            ca = res.find_atom("CA", "*")
            if ca is None:
                ca = res[0] if len(res) else None
            if ca is None:
                continue
            xyz.append([ca.pos.x, ca.pos.y, ca.pos.z])
            plddt.append(ca.b_iso)  # written as pLDDT*100 (write_result convention);
            # pDockQ2's fitted constants (x0=84.7) expect pLDDT on the AF-native
            # 0-100 scale, matching the af-analysis reference implementation -- do
            # NOT rescale to 0-1 here.
        out.append({"name": chain.name, "xyz": np.array(xyz), "plddt": np.array(plddt),
                    "n": len(xyz)})
    return out


def compute_metrics(cif_path, pae_npz_path, yaml_path, global_ptm):
    """Returns dict with pdockq2, ipsae, anticonf for the antigen-vs-antibody
    interface (antibody = all non-antigen chains combined), plus the raw inputs
    used, for a single (winner) structure."""
    yaml_chains = parse_yaml_chains(yaml_path)
    cif_chains = load_cif_chains(cif_path)
    if len(yaml_chains) != len(cif_chains):
        raise ValueError(f"chain count mismatch: yaml={len(yaml_chains)} cif={len(cif_chains)}")
    for (yid, yseq), c in zip(yaml_chains, cif_chains):
        if len(yseq) != c["n"]:
            raise ValueError(f"chain {yid}: yaml len={len(yseq)} cif residues={c['n']} -- "
                              "declaration-order assumption violated, cannot proceed")

    pae = np.load(pae_npz_path)["pae"]
    n_tok = pae.shape[0]
    offsets = np.cumsum([0] + [c["n"] for c in cif_chains])
    if offsets[-1] != n_tok:
        raise ValueError(f"token count mismatch: chains sum={offsets[-1]} pae shape={n_tok}")

    ag_idx = 0  # antigen is always the first declared chain in these yamls
    ag_lo, ag_hi = offsets[ag_idx], offsets[ag_idx + 1]
    ab_lo, ab_hi = offsets[1], offsets[-1]  # all remaining chains = antibody

    ag_xyz, ag_plddt = cif_chains[ag_idx]["xyz"], cif_chains[ag_idx]["plddt"]
    ab_xyz = np.concatenate([c["xyz"] for c in cif_chains[1:]], axis=0)
    ab_plddt = np.concatenate([c["plddt"] for c in cif_chains[1:]], axis=0)

    dist = np.sqrt(((ag_xyz[:, None, :] - ab_xyz[None, :, :]) ** 2).sum(-1))
    contact = dist < PDOCKQ2_CUTOFF_A
    ag_i, ab_j = np.where(contact)

    pae_block = pae[ag_lo:ag_hi, ab_lo:ab_hi]  # antigen(rows) x antibody(cols)
    pae_block_T = pae[ab_lo:ab_hi, ag_lo:ag_hi]

    if len(ag_i) == 0:
        pdockq2 = 0.0
        n_interface_res = 0
    else:
        pae_if = pae_block[ag_i, ab_j]
        norm_pae = np.mean(1.0 / (1.0 + (pae_if / PDOCKQ2_D0) ** 2))
        # interface residues on BOTH sides, matching af-analysis's per-chain-then-
        # averaged convention collapsed to one antigen-vs-antibody score
        plddt_if = np.concatenate([ag_plddt[np.unique(ag_i)], ab_plddt[np.unique(ab_j)]])
        x_val = norm_pae * float(np.mean(plddt_if))
        pdockq2 = PDOCKQ2_L / (1.0 + math.exp(-PDOCKQ2_K * (x_val - PDOCKQ2_X0))) + PDOCKQ2_B
        n_interface_res = len(np.unique(ag_i)) + len(np.unique(ab_j))

    # ipSAE: PAE-cutoff-filtered, TM-like score, both directions averaged (symmetric
    # chain-pair convention -- the DunbrackLab tool reports both directions; we
    # average them into one antigen<->antibody ipSAE for a single ranking number).
    L_pair = (ag_hi - ag_lo) + (ab_hi - ab_lo)
    d0 = calc_d0(L_pair)
    def _dir_ipsae(block):
        mask = block < IPSAE_PAE_CUTOFF
        if not mask.any():
            return 0.0
        ptm_vals = 1.0 / (1.0 + (block[mask] / d0) ** 2)
        return float(ptm_vals.mean())
    ipsae = 0.5 * (_dir_ipsae(pae_block) + _dir_ipsae(pae_block_T))

    anticonf = 0.7 * global_ptm + 0.3 * pdockq2

    return {"pdockq2": round(float(pdockq2), 6), "ipsae": round(float(ipsae), 6),
            "anticonf": round(float(anticonf), 6), "n_interface_residues": int(n_interface_res),
            "d0_ipsae": round(float(d0), 3)}


if __name__ == "__main__":
    import sys
    cif, pae_npz, yaml_path, ptm = sys.argv[1:5]
    print(json.dumps(compute_metrics(cif, pae_npz, yaml_path, float(ptm)), indent=2))
