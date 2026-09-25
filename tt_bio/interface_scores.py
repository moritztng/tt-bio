"""Interface scores from a folded complex: ipSAE, pDockQ, pDockQ2, LIS, ipTM and interface pAE.

These are arithmetic on the PAE matrix, the per-token pLDDT and the predicted coordinates, so they
need no device. The definitions follow `ipsae.py` as vendored by Adaptyv's Nipah competition
pipeline (github.com/adaptyvbio/nipah_ipsae_pipeline @ 80e0d56, Dunbrack's version 3 of
2025-04-06), which is the script the competition's ranking was computed with. Where that script
has a quirk, this module keeps the quirk, because a score that is "more correct" than the one a
participant is compared by is a different score:

- PAE rows are the aligned residue: `pae[i, j]` scores chain2 residue j with i's frame, i in chain1.
- A PAE pair counts when it is strictly below the cutoff (`<`); a contact counts at `<= 8 A`.
- Distances are CB-CB, CA for glycine, C3' for nucleotides. Tokens are CA atoms (C1' for
  nucleotides); ligand atoms and the non-backbone atoms of modified residues are masked out of
  the PAE and pLDDT arrays the way the script masks them.
- `n0dom` counts residue NUMBERS, not indices, per chain.
- The per-direction maxima are taken with numpy's first-index argmax over the whole residue axis.
- pLDDT is on the 0-100 scale. A missing pLDDT is zeros, as in the script, which drives pDockQ
  and pDockQ2 to their floor rather than failing.

Two outputs are not in the script and are defined here:

- `interface_pae`: mean PAE in Angstrom over both interchain blocks (chain1 x chain2 and
  chain2 x chain1), every pair, no cutoff. This is BindCraft's `i_pae` before its /31 scaling.
- `ipsae_min`: the smaller of the two directions. The Nipah notebook reports it as the "min" row
  beside the script's "max" row; the 2025 meta-analysis of 3,766 binders found it the better
  ranker, so both are returned and neither is chosen for the caller.

How this relates to the two other ipSAE computations in tt-bio:

- BoltzGen's design head (`boltzgen/model/layers/confidence_utils.py:compute_ipsae_score`, which
  design results report as `design_ipsae_min`) has this module's per-residue d0 and max over
  residues, at the same 15 A cutoff. Its d0 floor is 19 residues where the reference's is 27. On
  identical PAE the two agree to 6e-8 whenever the best residue has 27 or more partners under the
  cutoff, and differ by up to 0.0075 below that (400 random interfaces, rank correlation 0.9998;
  `tests/test_interface_scores.py`). The larger difference is the input: it scores BoltzGen's
  own refold with every target chain pooled, not a pinned Boltz-2 fold. It is a model-internal
  design score and stays as upstream BoltzGen wrote it. The canonical number is this module's.
- `scripts/abag_pae_metrics.py` computes a different quantity it also calls ipsae (see its
  docstring). It labels one finished research campaign and is not served.

The reference script takes its cutoffs as required arguments and has no default. Dunbrack's usage
examples pass 10/10; Adaptyv's competition notebook passes 15/15, and that is the call this module
matches.

`SCORES_VERSION` names these definitions. Anything that changes a number for the same inputs
must bump it: a stored score carries the version it was computed under, so an old number is
never silently re-meant.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SCORES_VERSION = "ipsae-v3-nipah80e0d56.1"
REFERENCE = "github.com/adaptyvbio/nipah_ipsae_pipeline@80e0d56 ipsae.py (Dunbrack v3, 2025-04-06)"
# The cutoffs the Nipah notebook calls the script with (`calculate_ipsae(..., 15.0, 15.0)`).
# Dunbrack's paper recommends 10/10; the competition used 15/15, and matching the competition is
# the point. dist_cutoff only moves the residue counts, never a score.
PAE_CUTOFF = 15.0
DIST_CUTOFF = 15.0
CONTACT_A = 8.0

_RESIDUES = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
             "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
             "DA", "DC", "DT", "DG", "A", "C", "U", "G"}
_NUCLEOTIDES = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}


def _d0(n, nucleic: bool):
    """TM-score d0 of Yang and Skolnick (2004) with the script's floor of 27 residues."""
    n = np.maximum(27.0, np.asarray(n, dtype=float))
    return np.maximum(2.0 if nucleic else 1.0, 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8)


def _ptm(pae, d0):
    return 1.0 / (1.0 + (pae / d0) ** 2.0)


def read_cif_tokens(path) -> dict:
    """Tokens of a Boltz-style mmCIF, parsed the way the reference script parses it.

    Returns chain ids, residue numbers and names per token, the CB-like coordinate per token, and
    `token_mask`, the selector from the model's full token axis (which includes ligand atoms) down
    to the residue tokens the scores are defined on."""
    fields: dict[str, int] = {}
    chains, resnums, resnames, cb, mask = [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("_atom_site."):
            fields[line.strip().split(".", 1)[1]] = len(fields)
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        f = line.split()
        name, res, seq = f[fields["label_atom_id"]], f[fields["label_comp_id"]], f[fields["label_seq_id"]]
        if seq == ".":                      # a ligand atom is one token, never scored
            mask.append(0)
            continue
        xyz = [float(f[fields[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")]
        if name == "CA" or "C1" in name:
            mask.append(1)
            chains.append(f[fields["label_asym_id"]])
            resnums.append(int(seq))
            resnames.append(res)
        if name == "CB" or "C3" in name or (res == "GLY" and name == "CA"):
            cb.append(xyz)
        if name != "CA" and "C1" not in name and res not in _RESIDUES:
            mask.append(0)                  # extra atoms of a modified residue are tokens too
    if len(cb) != len(chains):
        raise ValueError(f"{path}: {len(chains)} residue tokens but {len(cb)} CB-like atoms; "
                         "a residue without CB (or CA for glycine) cannot be scored")
    return {"chains": np.array(chains), "resnums": np.array(resnums), "resnames": np.array(resnames),
            "cb": np.array(cb, dtype=float), "token_mask": np.array(mask, dtype=bool)}


def score(pae, plddt, chains, cb, resnums=None, resnames=None, pair_iptm=None,
          pae_cutoff: float = PAE_CUTOFF, dist_cutoff: float = DIST_CUTOFF) -> dict:
    """Every chain pair's interface scores.

    `pae` is (N, N) in Angstrom over the N residue tokens, `plddt` (N,) on 0-100, `chains` (N,)
    chain ids, `cb` (N, 3). `pair_iptm[c1][c2]` is the model's own chain-pair ipTM if it has one.
    Returns `{"pairs": {"A-B": {...}}, "directions": {"A->B": {...}}}` with each unordered pair
    once, keyed in sorted order."""
    pae = np.asarray(pae, dtype=float)
    plddt = np.zeros(len(chains)) if plddt is None else np.asarray(plddt, dtype=float)
    chains = np.asarray(chains)
    resnums = np.arange(len(chains)) if resnums is None else np.asarray(resnums)
    n = len(chains)
    if pae.shape != (n, n) or plddt.shape != (n,) or np.shape(cb) != (n, 3):
        raise ValueError(f"shape mismatch: pae {pae.shape}, plddt {plddt.shape}, cb {np.shape(cb)}, "
                         f"{n} tokens")
    cb = np.asarray(cb, dtype=float)
    dist = np.sqrt(((cb[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
    ids = sorted(set(chains.tolist()))
    nucleic = {c: resnames is not None and any(r in _NUCLEOTIDES for r in np.asarray(resnames)[chains == c])
               for c in ids}

    direc = {}
    for c1 in ids:
        for c2 in ids:
            if c1 == c2:
                continue
            na = nucleic[c1] or nucleic[c2]
            r1, r2 = chains == c1, chains == c2
            valid = r2[None, :] & (pae < pae_cutoff)          # (N, N), every row, as the script
            rows = np.where(r1)[0]

            def by_res(ptm_rows, sel):
                """Row means over `sel`, zero where a row has no pair, on the full N axis."""
                out = np.zeros(n)
                cnt = sel[rows].sum(1)
                s = np.where(sel[rows], ptm_rows, 0.0).sum(1)
                out[rows] = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
                return out

            d0chn = float(_d0(r1.sum() + r2.sum(), na))
            ptm_chn = _ptm(pae[rows], d0chn)
            iptm_chn = by_res(ptm_chn, np.broadcast_to(r2[None, :], (n, n)))
            ipsae_chn = by_res(ptm_chn, valid)

            hit_rows = rows[valid[rows].any(1)]
            n0dom = len(set(resnums[hit_rows].tolist())) + len(set(resnums[valid[rows].any(0)].tolist()))
            d0dom = float(_d0(n0dom, na))
            ipsae_dom = by_res(_ptm(pae[rows], d0dom), valid)

            n0res_all = valid.sum(1)
            d0res_all = _d0(n0res_all, na)
            ipsae_res = by_res(_ptm(pae[rows], d0res_all[rows][:, None]), valid)

            contact = r1[:, None] & r2[None, :] & (dist <= CONTACT_A)
            npairs = int(contact.sum())
            if npairs:
                members = contact.any(1) | contact.any(0)
                mean_plddt = plddt[members].mean()
                pdockq = 0.724 / (1 + math.exp(-0.052 * (mean_plddt * math.log10(npairs) - 152.611))) + 0.018
                mean_ptm = _ptm(pae[contact], 10.0).sum() / npairs
                pdockq2 = 1.31 / (1 + math.exp(-0.075 * (mean_plddt * mean_ptm - 84.733))) + 0.005
            else:
                pdockq = pdockq2 = 0.0

            block = pae[np.ix_(r1, r2)]
            lis_vals = block[block <= 12]
            lis = float(((12 - lis_vals) / 12).mean()) if lis_vals.size else 0.0

            dist_ok = valid & (dist < dist_cutoff)
            k = int(np.argmax(ipsae_res))
            direc[(c1, c2)] = {
                "ipsae": float(ipsae_res[k]),
                "ipsae_d0chn": float(ipsae_chn.max()),
                "ipsae_d0dom": float(ipsae_dom.max()),
                "iptm_d0chn": float(iptm_chn.max()),
                "iptm": None if pair_iptm is None else float(pair_iptm[c1][c2]),
                "pdockq": float(pdockq), "pdockq2": float(pdockq2), "lis": lis,
                "n0res": int(n0res_all[k]), "d0res": float(d0res_all[k]),
                "n0chn": int(r1.sum() + r2.sum()), "d0chn": d0chn, "n0dom": int(n0dom), "d0dom": d0dom,
                "nres1": len(set(resnums[hit_rows].tolist())),
                "nres2": len(set(resnums[valid[rows].any(0)].tolist())),
                "dist1": len(set(resnums[rows[dist_ok[rows].any(1)]].tolist())),
                "dist2": len(set(resnums[dist_ok[rows].any(0)].tolist())),
                "_block_pae_sum": float(block.sum()), "_block_n": int(block.size),
            }

    pairs = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ab, ba = direc[(a, b)], direc[(b, a)]
            iptm = None if ab["iptm"] is None else max(ab["iptm"], ba["iptm"])
            pairs[f"{a}-{b}"] = {
                "ipsae": max(ab["ipsae"], ba["ipsae"]),
                "ipsae_min": min(ab["ipsae"], ba["ipsae"]),
                "ipsae_d0chn": max(ab["ipsae_d0chn"], ba["ipsae_d0chn"]),
                "ipsae_d0dom": max(ab["ipsae_d0dom"], ba["ipsae_d0dom"]),
                "iptm": iptm,
                "iptm_d0chn": max(ab["iptm_d0chn"], ba["iptm_d0chn"]),
                "pdockq": ab["pdockq"],
                "pdockq2": max(ab["pdockq2"], ba["pdockq2"]),
                "lis": (ab["lis"] + ba["lis"]) / 2.0,
                "interface_pae": (ab["_block_pae_sum"] + ba["_block_pae_sum"]) / (ab["_block_n"] + ba["_block_n"]),
            }
    for d in direc.values():
        d.pop("_block_pae_sum"), d.pop("_block_n")
    return {"version": SCORES_VERSION, "pae_cutoff": pae_cutoff, "dist_cutoff": dist_cutoff,
            "pairs": pairs, "directions": {f"{a}->{b}": v for (a, b), v in direc.items()}}


def score_files(structure, pae, plddt=None, confidence=None,
                pae_cutoff: float = PAE_CUTOFF, dist_cutoff: float = DIST_CUTOFF,
                pair_iptm=None) -> dict:
    """Score a fold from its files: an mmCIF, a PAE `.npz` (key `pae`), optionally a pLDDT `.npz`
    (key `plddt`, 0-1 as Boltz writes it) and a confidence JSON carrying `pair_chains_iptm`.
    Accepts tt-bio's and upstream Boltz's file layouts alike. Arrays may be passed in place of
    the `.npz` paths.

    The confidence JSON's matrix is indexed by chain declaration order, and the reference script
    maps a chain letter to it by alphabet position (A=0, B=1). That swaps the two directions of a
    binder declared as B before a target declared as A, which the max over both directions hides.
    A caller that knows the real order passes `pair_iptm` as `{chain: {chain: iptm}}` instead."""
    tok = read_cif_tokens(structure)
    m = tok["token_mask"]

    def arr(x, key):
        return np.load(x)[key] if isinstance(x, (str, Path)) else np.asarray(x)

    pae_full = arr(pae, "pae")
    if pae_full.shape != (len(m), len(m)):
        raise ValueError(f"PAE is {pae_full.shape} but {structure} has {len(m)} tokens")
    pl = None if plddt is None else 100.0 * arr(plddt, "plddt")[m]
    iptm = pair_iptm
    if iptm is None and confidence is not None:
        c = json.loads(Path(confidence).read_text()) if isinstance(confidence, (str, Path)) else confidence
        pci = c.get("pair_chains_iptm")
        if pci:
            # Chain letters map to the matrix index by position in the alphabet, as the script does.
            ids = sorted(set(tok["chains"].tolist()))
            iptm = {a: {b: pci[str(ord(a) - 65)][str(ord(b) - 65)] for b in ids if b != a} for a in ids}
    return score(pae_full[np.ix_(m, m)], pl, tok["chains"], tok["cb"], tok["resnums"],
                 tok["resnames"], iptm, pae_cutoff, dist_cutoff)


# The metrics a distribution is summarised for. The counts and d0 values are bookkeeping.
DISTRIBUTION_KEYS = ("ipsae", "ipsae_min", "ipsae_d0chn", "ipsae_d0dom", "iptm", "iptm_d0chn",
                     "pdockq", "pdockq2", "lis", "interface_pae")


def distribution(samples: list[dict]) -> dict:
    """Summarise per-sample `score` outputs of one input into a distribution per chain pair.

    For each pair and metric: every value in sample order, their mean, the sample standard
    deviation (ddof=1, None for a single sample) and the range. The standard deviation of a score
    is its stability figure. Nothing is collapsed: the values stay in the output, so a caller can
    compute any other statistic without refolding.

    Boltz-2's only stochastic step at inference is the diffusion noise (featurization draws from a
    fixed generator, and dropout is off), so the diffusion samples of one fold and the same number
    of separately seeded folds draw from the same distribution. The samples share one trunk pass
    and cost a fraction of the folds."""
    if not samples:
        raise ValueError("no samples to summarise")
    out = {}
    for pair in samples[0]["pairs"]:
        out[pair] = {}
        for key in DISTRIBUTION_KEYS:
            vals = [s["pairs"][pair][key] for s in samples]
            if any(v is None for v in vals):
                continue
            a = np.asarray(vals, dtype=float)
            out[pair][key] = {"values": [float(v) for v in a], "mean": float(a.mean()),
                              "sd": float(a.std(ddof=1)) if a.size > 1 else None,
                              "min": float(a.min()), "max": float(a.max())}
    return {"version": SCORES_VERSION, "n": len(samples), "pairs": out}
