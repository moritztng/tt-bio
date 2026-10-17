"""Boltz-2's relative-position joint table reproduces the encoder bit for bit (host only)."""
import torch

from tt_bio.boltz2 import RelativePositionEncoder


def _feats(n, seed):
    g = torch.Generator().manual_seed(seed)
    # Three chains, the last two copies of one entity, with a ligand-like run of shared residues.
    asym = torch.cat([torch.zeros(n // 2), torch.ones(n // 4), torch.full((n - n // 2 - n // 4,), 2)]).long()
    entity = torch.where(asym == 0, 0, 1)
    sym = torch.where(asym == 2, 1, 0)
    residue = torch.cat([torch.arange(n // 2), torch.arange(n - n // 2)])
    residue[n // 2 - 8:n // 2] = residue[n // 2 - 8]  # one residue of eight tokens
    residue += torch.randint(0, 3, (n,), generator=g).cumsum(0)  # gaps
    token = torch.arange(n)
    return {k: v[None] for k, v in dict(asym_id=asym, entity_id=entity, sym_id=sym,
                                        residue_index=residue, token_index=token).items()}


def test_joint_table_is_forward_bit_for_bit():
    torch.manual_seed(0)
    for fix_sym_check in (False, True):
        enc = RelativePositionEncoder(128, fix_sym_check=fix_sym_check)
        for n, seed in ((37, 0), (200, 1)):
            feats = _feats(n, seed)
            with torch.no_grad():
                want = enc(feats)
                index, table = enc.joint_table(feats)
            assert index.dtype == torch.int32 and int(index.min()) == 1
            assert torch.equal(table[0], torch.zeros(128))
            got = table[index.long()]
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))
            assert table.shape[0] < 1000
