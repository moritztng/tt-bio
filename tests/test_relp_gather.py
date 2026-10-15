"""relp_gather's host half: gathering the per-fold table reproduces the one-hot projection.

The device matmul it replaces multiplies a 0/1 one-hot by bf16 weights, so the exact answer is the
fp32 sum of at most four bf16 weight rows. These checks hold the table to that sum, bit for bit,
on a multi-chain input with a token-bucket pad."""
import types

import torch

from tt_bio.protenix import TrunkInput, Protenix


def _feats(n, pad):
    asym = torch.tensor([0] * 40 + [1] * 50 + [2] * (n - 90))
    feats = dict(asym_id=asym, residue_index=torch.cat([torch.arange(40), torch.arange(50), torch.arange(n - 90)]),
                 entity_id=torch.tensor([0] * 40 + [0] * 50 + [1] * (n - 90)),
                 token_index=torch.arange(n), sym_id=torch.tensor([0] * 40 + [1] * 50 + [0] * (n - 90)))
    relp = Protenix._generate_relp(feats)
    return torch.nn.functional.pad(relp, (0, 0, 0, pad, 0, pad)) if pad else relp


def test_table_gather_equals_one_hot_projection():
    torch.manual_seed(0)
    w = torch.randn(128, 139)
    fake = types.SimpleNamespace(_w={"relative_position_encoding.linear_no_bias.weight": w},
                                 _RELP_CODE=TrunkInput._RELP_CODE)
    relp = _feats(150, 10)
    index, table = TrunkInput.relp_index(fake, relp)
    ref = (relp.float() @ w.to(torch.bfloat16).float().t()).to(torch.bfloat16)
    assert index.shape == relp.shape[:-1] and index.dtype == torch.int32
    assert torch.equal(table[index.long()], ref)
    assert torch.equal(table[index[-1, -1].long()], torch.zeros(128, dtype=torch.bfloat16))
