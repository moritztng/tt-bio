"""Protenix / Mol-1 confidence outputs in upstream's exact form.

A port of ``mol/model/sample_confidence.py`` and ``mol/runner/dumper.py`` (Mol-1 @ e6e64866, both unchanged from
Protenix @ bd54a05): the summary-confidence and full-data dicts computed from the confidence head's logits, the
ranking, and the file layout

    <out_dir>/<job>/seed_<s>/predictions/<job>_sample_<k>.cif
                                         <job>_summary_confidence_sample_<k>.json
                                         <job>_full_data_sample_<k>.json      (need_atom_confidence)

where k is the sample's rank by ``ranking_score`` within the seed, 0 = best.

This is upstream's contract, not the family rule in ``tt_bio.ranking``: ``ranking_score`` is AF3's
0.8 ipTM + 0.2 pTM + 0.5 disorder - 100 has_clash for every target, including monomers, and every ``chain_pair_*``
diagonal is 0.0. A run given upstream JSON input reports this, so its files read the same as upstream's.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# mol/config/base.py: loss.{plddt,pde,pae,distogram}, metrics.clash
PLDDT_BINS = (0.0, 1.0, 50)
PDE_BINS = PAE_BINS = (0.0, 32.0, 64)
DISTOGRAM_BINS = (2.3125, 21.6875, 64)
AF3_CLASH_THRESHOLD = 1.1


def bin_centers(min_bin, max_bin, no_bins):
    w = (max_bin - min_bin) / no_bins
    return torch.linspace(min_bin, max_bin - w, no_bins) + 0.5 * w


def logits_to_score(logits, bins, return_prob=False):
    prob = torch.softmax(logits, dim=-1)
    score = prob @ bin_centers(*bins).to(logits.device)
    return (score, prob) if return_prob else score


def contact_probs(distogram_logits, thres=8.0):
    """[N_tok, N_tok, 64] distogram logits -> P(distance < 8 A) (upstream ``compute_contact_prob``)."""
    prob = torch.softmax(distogram_logits, dim=-1)
    k = int((bin_centers(*DISTOGRAM_BINS) < thres).sum())
    return prob[..., :k].sum(-1)


def _norm(n):
    return 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8


def _tm_weight(n, bins):
    return 1 / (1 + (bin_centers(*bins) / _norm(n)) ** 2)


def calculate_ptm(pae_prob, has_frame, token_mask=None):
    has_frame = has_frame.bool()
    if token_mask is not None:
        m = token_mask.bool()
        pae_prob, has_frame = pae_prob[..., m, :, :][..., :, m, :], has_frame[m]
    if has_frame.sum() == 0:
        return torch.zeros(pae_prob.shape[:-3])
    tt = (pae_prob * _tm_weight(has_frame.shape[-1], PAE_BINS)).sum(-1)
    return tt.mean(-1)[..., has_frame].max(-1).values


def calculate_iptm(pae_prob, has_frame, asym_id, token_mask=None, eps=1e-8):
    has_frame = has_frame.bool()
    if token_mask is not None:
        m = token_mask.bool()
        pae_prob, has_frame, asym_id = pae_prob[..., m, :, :][..., :, m, :], has_frame[m], asym_id[m]
    if has_frame.sum() == 0:
        return torch.zeros(pae_prob.shape[:-3])
    tt = (pae_prob * _tm_weight(has_frame.shape[-1], PAE_BINS)).sum(-1)
    diff = asym_id[None, :] != asym_id[:, None]
    iptm = (tt * diff).sum(-1) / (eps + diff.sum(-1))
    return iptm[..., has_frame].max(-1).values


def _dense(asym_id):
    """asym ids renumbered 0..n_chain-1 in sorted order (upstream remaps the same way)."""
    asym_id = asym_id.long()
    return torch.unique(asym_id, return_inverse=True)[1]


def chain_ptm(pae_prob, has_frame, asym_id, token_is_ligand):
    has_frame, a = has_frame.bool(), _dense(asym_id)
    n = int(a.max()) + 1
    masks = [a == c for c in range(n)]
    is_lig = [token_is_ligand[m].sum() >= m.sum() // 2 for m in masks]
    B = pae_prob.shape[:-3]
    pair = torch.zeros(B + (n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pair[:, i, j] = pair[:, j, i] = calculate_iptm(pae_prob, has_frame, a, token_mask=masks[i] | masks[j])
    cptm = torch.zeros(B + (n,))
    for c in range(n):
        cptm[:, c] = calculate_ptm(pae_prob, has_frame, token_mask=masks[c])
    framed = [bool((masks[i] & has_frame).any()) for i in range(n)]
    ciptm = torch.zeros(B + (n,))
    for c in range(n):
        vals = [pair[:, i, j] for i in range(n) for j in range(n) if (i == c or j == c) and i != j and framed[i]]
        if vals:
            ciptm[:, c] = torch.stack(vals, -1).mean(-1)
    glob = torch.zeros(B + (n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            glob[:, i, j] = (ciptm[:, i] if is_lig[i] else ciptm[:, j] if is_lig[j]
                             else (ciptm[:, i] + ciptm[:, j]) * 0.5)
    return {"chain_ptm": cptm, "chain_iptm": ciptm, "chain_pair_iptm": pair, "chain_pair_iptm_global": glob}


def chain_gpde(pde, contact, asym_id, eps=1e-8):
    a = _dense(asym_id)
    n = int(a.max()) + 1
    B = pde.shape[:-2]

    def g(m1, m2):
        c, p = contact[..., m1, :][..., m2], pde[..., m1, :][..., m2]
        return (p * c).sum((-1, -2)) / (c.sum((-1, -2)) + eps)

    cg = torch.zeros(B + (n,))
    for c in range(n):
        cg[..., c] = g(a == c, a == c)
    pg = torch.zeros(B + (n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pg[..., i, j] = pg[..., j, i] = g(a == i, a == j)
    return {"chain_gpde": cg, "chain_pair_gpde": pg}


def chain_plddt(atom_plddt, asym_id, atom_to_token_idx):
    a = _dense(asym_id)
    n = int(a.max()) + 1
    B = atom_plddt.shape[:-1]
    sub = lambda tm: atom_plddt[:, tm[atom_to_token_idx]].mean(-1)
    cp = torch.zeros(B + (n,))
    for c in range(n):
        cp[:, c] = sub(a == c)
    pp = torch.zeros(B + (n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                pp[:, i, j] = sub((a == i) | (a == j))
    return {"chain_plddt": cp, "chain_pair_plddt": pp}


def has_clash(coords, asym_id, atom_to_token_idx, atom_is_polymer, threshold=AF3_CLASH_THRESHOLD):
    """[N_s, N_atom, 3] -> [N_s] bool. AF3 clash: two polymer chains with > 100 atom pairs closer than 1.1 A, or
    more than half of the smaller chain's atom count. Ligand chains never count. As upstream wires it, DNA and RNA
    count as polymer and a chain's type comes from its atoms' polymer flag."""
    a = _dense(asym_id)
    n = int(a.max()) + 1
    masks = [(a == c)[atom_to_token_idx] for c in range(n)]
    poly = []
    for m in masks:
        t = atom_is_polymer[m].long().unique()
        assert len(t) == 1, "a chain mixes polymer and ligand atoms"
        poly.append(bool(t[0]))
    out = torch.zeros(coords.shape[0], dtype=torch.bool)
    for s in range(coords.shape[0]):
        for i in range(n):
            for j in range(i + 1, n):
                if not (poly[i] and poly[j]):
                    continue
                total = int((torch.cdist(coords[s][masks[i]], coords[s][masks[j]]) < threshold).sum())
                rel = total / min(int(masks[i].sum()), int(masks[j].sum()))
                out[s] |= total > 100 or rel > 0.5
    return out


def summarize(pae_logits, plddt_logits, pde_logits, contact, token_asym_id, token_has_frame, coords,
              atom_to_token_idx, atom_is_polymer, n_recycle):
    """Upstream ``compute_full_data_and_summary`` for N_s samples: lists of per-sample summary and full-data dicts.

    pae/pde logits [N_s, N_tok, N_tok, 64], plddt logits [N_s, N_atom, 50], contact [N_tok, N_tok] or
    [N_s, N_tok, N_tok], coords [N_s, N_atom, 3]. Computed per sample, in float32, like upstream. Full data is
    always built (upstream's model calls this with return_full_data=True): it carries the B-factors."""
    n_s = pae_logits.shape[0]
    if contact.dim() == 2:
        contact = contact.unsqueeze(0).expand(n_s, -1, -1)
    atom_to_token_idx = atom_to_token_idx.long()
    atom_is_polymer = atom_is_polymer.long()
    tok_lig = torch.zeros_like(token_asym_id.long()).scatter_add(0, atom_to_token_idx, 1 - atom_is_polymer) > 0
    summaries, fulls = [], []
    for i in range(n_s):
        fd = {"atom_plddt": logits_to_score(plddt_logits[i:i + 1].float(), PLDDT_BINS),
              "token_pair_pde": logits_to_score(pde_logits[i:i + 1].float(), PDE_BINS),
              "contact_probs": contact[i].float().clone()}
        fd["token_pair_pae"], pae_prob = logits_to_score(pae_logits[i:i + 1].float(), PAE_BINS, return_prob=True)
        sm = {"plddt": fd["atom_plddt"].mean(-1) * 100,
              "gpde": (fd["token_pair_pde"] * fd["contact_probs"]).sum((-1, -2)) / fd["contact_probs"].sum((-1, -2)),
              "ptm": calculate_ptm(pae_prob, token_has_frame),
              "iptm": calculate_iptm(pae_prob, token_has_frame, token_asym_id)}
        sm.update(chain_gpde(fd["token_pair_pde"], fd["contact_probs"], token_asym_id))
        sm.update(chain_ptm(pae_prob, token_has_frame, token_asym_id, tok_lig))
        sm.update(chain_plddt(fd["atom_plddt"], token_asym_id, atom_to_token_idx))
        sm["has_clash"] = has_clash(coords[i:i + 1].float(), token_asym_id, atom_to_token_idx, atom_is_polymer)
        sm["disorder"] = torch.zeros_like(sm["ptm"])
        sm["ranking_score"] = 0.8 * sm["iptm"] + 0.2 * sm["ptm"] + 0.5 * sm["disorder"] - 100 * sm["has_clash"]
        s = {k: v[0] for k, v in sm.items()}
        s["num_recycles"] = torch.tensor(n_recycle)
        summaries.append(s)
        f = {k: v[0] for k, v in fd.items() if k != "contact_probs"}
        f.update(contact_probs=fd["contact_probs"], token_has_frame=token_has_frame.clone(),
                 token_asym_id=token_asym_id.clone(), atom_to_token_idx=atom_to_token_idx.clone())
        fulls.append(f)
    return summaries, fulls


def rank_of(summaries):
    """Sample index -> output number k, by ranking_score descending (upstream's argsort of argsort)."""
    v = torch.tensor([float(s["ranking_score"]) for s in summaries])
    return [int(r) for r in torch.argsort(torch.argsort(v, descending=True, stable=True), stable=True)]


def _jsonable(d, round2=False):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.float().cpu().numpy() if v.dtype == torch.bfloat16 else v.cpu().numpy()
        if round2 and isinstance(v, np.ndarray):
            v = np.round(v, 2)
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def predictions_dir(out_dir, job, seed) -> Path:
    return Path(out_dir) / job / f"seed_{seed}" / "predictions"


def write(out_dir, job, seed, summaries, fulls, write_structure=None, need_atom_confidence=False):
    """Write one seed's confidence files (and structures via ``write_structure(sample_idx, path, b_factor)``,
    b_factor = atom pLDDT x 100 rounded to 2 decimals) under upstream's names. Summary values are written
    unrounded and full-data arrays (only with need_atom_confidence) rounded to 2 decimals, both exactly as
    upstream's ``DataDumper``. Returns the rank of each sample."""
    d = predictions_dir(out_dir, job, seed)
    d.mkdir(parents=True, exist_ok=True)
    ranks = rank_of(summaries)
    for i, k in enumerate(ranks):
        if write_structure is not None:
            bf = np.round(fulls[i]["atom_plddt"].float().cpu().numpy() * 100.0, 2)
            write_structure(i, d / f"{job}_sample_{k}.cif", bf)
        with open(d / f"{job}_summary_confidence_sample_{k}.json", "w") as fh:
            json.dump(_jsonable(summaries[i]), fh, indent=4)
        if need_atom_confidence:
            with open(d / f"{job}_full_data_sample_{k}.json", "w") as fh:
                json.dump(_jsonable(fulls[i], round2=True), fh)
    return ranks
