"""derive_relpos_codes is derive_relpos exactly: table[codes] equals the dense one-hot feature."""
import torch

from tt_bio.openfold3_host_prep import derive_relpos, derive_relpos_codes


def _feat(chains, cyclic=None, homomer=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    asym, ent, sym, res, tok = [], [], [], [], []
    t = 0
    for c, n in enumerate(chains):
        r = torch.arange(n)
        if c == len(chains) - 1 and n > 4:          # a ligand-like chain: atomized tokens share a residue
            r = torch.zeros(n, dtype=torch.long)
        asym += [c] * n; ent += [0 if homomer else c] * n; sym += [c if homomer else 0] * n
        res.append(r + int(torch.randint(0, 5, (1,), generator=g))); tok.append(torch.arange(t, t + n)); t += n
    f = dict(asym_id=torch.tensor(asym), entity_id=torch.tensor(ent), sym_id=torch.tensor(sym),
             residue_index=torch.cat(res), token_index=torch.cat(tok))
    if cyclic is not None:
        f["cyclic_mask"] = torch.tensor(asym) == cyclic
    return f


def test_codes_match_dense():
    for f in (_feat([40]), _feat([30, 30], homomer=True), _feat([25, 50, 7]), _feat([20, 30, 6], cyclic=0),
              _feat([90, 3], seed=3)):
        codes, table = derive_relpos_codes(f)
        dense = derive_relpos(f)
        assert codes.dtype == torch.int32 and table.shape[-1] == dense.shape[-1] == 139
        assert torch.equal(table[codes.long()], dense)
