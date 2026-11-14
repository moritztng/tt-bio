"""Golden for tests/test_protenix_output.py: Mol-1's own compute_full_data_and_summary on small synthetic inputs.

    /home/moritz/mol1-upstream/io-venv/bin/python -I scripts/mol1_io/confidence_goldens.py \
        tests/data/mol1_io/confidence_golden.pt

Inputs are seeded random logits and coordinates for three layouts (monomer; protein dimer + ligand; a dimer whose
chains overlap so has_clash fires). Writes inputs and upstream outputs to one small torch file.
"""
import sys

import torch
from ml_collections.config_dict import ConfigDict
from mol.model.sample_confidence import compute_contact_prob, compute_full_data_and_summary

CFG = ConfigDict({"loss": {"plddt": {"min_bin": 0, "max_bin": 1.0, "no_bins": 50},
                           "pde": {"min_bin": 0, "max_bin": 32, "no_bins": 64},
                           "pae": {"min_bin": 0, "max_bin": 32, "no_bins": 64}},
                  "metrics": {"clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75}}})


def case(g, chains, n_s=3, clash=False):
    """chains: list of (n_tokens, atoms_per_token, is_polymer)."""
    asym, a2t, poly, t = [], [], [], 0
    for c, (nt, apt, p) in enumerate(chains):
        for _ in range(nt):
            asym.append(c); a2t += [t] * apt; poly += [int(p)] * apt; t += 1
    nt, na = t, len(a2t)
    coords = torch.randn(n_s, na, 3, generator=g) * 8
    if clash:   # pile the second chain onto the first
        a2t_t, asym_t = torch.tensor(a2t), torch.tensor(asym)
        m0, m1 = asym_t[a2t_t] == 0, asym_t[a2t_t] == 1
        k = int(min(m0.sum(), m1.sum()))
        coords[0, torch.where(m1)[0][:k]] = coords[0, torch.where(m0)[0][:k]] + 0.1
    dist = torch.randn(nt, nt, 64, generator=g)
    return dict(pae_logits=torch.randn(n_s, nt, nt, 64, generator=g) * 3,
                plddt_logits=torch.randn(n_s, na, 50, generator=g) * 3,
                pde_logits=torch.randn(n_s, nt, nt, 64, generator=g) * 3,
                distogram_logits=dist,
                contact_probs=compute_contact_prob(dist, 2.3125, 21.6875, 64),
                token_asym_id=torch.tensor(asym), token_has_frame=(torch.rand(nt, generator=g) > 0.1).long(),
                atom_coordinate=coords, atom_to_token_idx=torch.tensor(a2t), atom_is_polymer=torch.tensor(poly),
                N_recycle=20)


def main(out):
    g = torch.Generator().manual_seed(0)
    cases = {"monomer": case(g, [(12, 3, True)]),
             "dimer_ligand": case(g, [(10, 3, True), (8, 3, True), (6, 1, False)]),
             "clash": case(g, [(9, 4, True), (9, 4, True)], n_s=2, clash=True)}
    for c in cases.values():
        inp = {k: v for k, v in c.items() if k != "distogram_logits"}
        c["summary"], c["full"] = compute_full_data_and_summary(CFG, **inp, return_full_data=True)
    torch.save(cases, out)


if __name__ == "__main__":
    main(sys.argv[1])
