"""BoltzGen's RelativePositionEncoder is Boltz-2's read through BoltzGen's feature names.

The device z_init gathers ``joint_table``, so it must equal ``forward`` exactly, and both must
read ``feature_asym_id``/``feature_residue_index``/``cyclic``, never Boltz-2's keys.
"""
import torch
from torch.nn.functional import one_hot

from tt_bio.boltzgen.model.modules.encoders import RelativePositionEncoder


def _feats(cyclic):
    asym = torch.cat([torch.zeros(40), torch.ones(30), torch.full((20,), 2)]).long()[None]
    res = torch.cat([torch.arange(40), torch.arange(30), torch.arange(20)])[None]
    res[0, 10:13] = 9
    cyc = torch.zeros(1, 90, dtype=torch.long)
    if cyclic:
        cyc[0, 40:70] = 30
    return {"feature_asym_id": asym, "feature_residue_index": res, "token_index": torch.arange(90)[None],
            "entity_id": torch.cat([torch.zeros(70), torch.ones(20)]).long()[None],
            "sym_id": torch.cat([torch.zeros(40), torch.ones(30), torch.zeros(20)]).long()[None],
            "cyclic": cyc, "asym_id": asym * 0, "residue_index": res * 0}


def _reference(m, f):
    """BoltzGen's original Algorithm 3, one-hot then project."""
    r, s = m.r_max, m.s_max
    same_chain = f["feature_asym_id"][:, :, None] == f["feature_asym_id"][:, None, :]
    same_res = f["feature_residue_index"][:, :, None] == f["feature_residue_index"][:, None, :]
    same_ent = f["entity_id"][:, :, None] == f["entity_id"][:, None, :]
    d_res = f["feature_residue_index"][:, :, None] - f["feature_residue_index"][:, None, :]
    if torch.any(f["cyclic"] > 0):
        period = torch.where(f["cyclic"] > 0, f["cyclic"], torch.zeros_like(f["cyclic"]) + 10000).unsqueeze(1)
        d_res = (d_res - period * torch.round(d_res / period)).long()
    d_res = torch.where(same_chain, torch.clip(d_res + r, 0, 2 * r), 2 * r + 1)
    d_tok = torch.clip(f["token_index"][:, :, None] - f["token_index"][:, None, :] + r, 0, 2 * r)
    d_tok = torch.where(same_chain & same_res, d_tok, 2 * r + 1)
    d_ch = torch.clip(f["sym_id"][:, :, None] - f["sym_id"][:, None, :] + s, 0, 2 * s)
    d_ch = torch.where(~same_ent, 2 * s + 1, d_ch)
    return m.linear_layer(torch.cat([one_hot(d_res, 2 * r + 2).float(), one_hot(d_tok, 2 * r + 2).float(),
                                     same_ent.unsqueeze(-1).float(), one_hot(d_ch, 2 * s + 2).float()], -1))


def test_forward_and_joint_table_match_the_boltzgen_reference():
    torch.manual_seed(0)
    m = RelativePositionEncoder(32)
    for cyclic in (False, True):
        f = _feats(cyclic)
        ref = _reference(m, f)
        assert torch.equal(m(f), ref)
        index, table = m.joint_table(f)
        assert torch.equal(table[index.long()], ref)
