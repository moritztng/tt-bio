"""The template feature embedder projects restype_ti / restype_tj at the row they broadcast."""
import torch

from tt_bio.openfold3_host_prep import derive_template_feat
from tt_bio.openfold3_template import _unbroadcast


def _feat(n=7, nt=2):
    g = torch.Generator().manual_seed(0)
    restype = torch.nn.functional.one_hot(torch.randint(0, 32, (nt, n), generator=g), 32).float()
    return derive_template_feat({
        "asym_id": torch.tensor([0] * 4 + [1] * (n - 4)),
        "template_pseudo_beta_mask": torch.ones(nt, n),
        "template_backbone_frame_mask": torch.ones(nt, n),
        "template_restype": restype,
        "template_unit_vector": torch.randn(nt, n, n, 3, generator=g),
        "template_distogram": torch.zeros(nt, n, n, 39),
    })


def test_restype_rows_broadcast_back_to_the_feature():
    f = _feat()
    for key, axis in (("restype_ti", -2), ("restype_tj", -3)):
        r = _unbroadcast(key, f[key])
        assert r.shape[axis] == 1
        assert torch.equal(r.expand_as(f[key]), f[key])


def test_other_features_and_non_broadcast_tensors_are_kept_whole():
    f = _feat()
    assert _unbroadcast("unit_vec_x", f["unit_vec_x"]) is f["unit_vec_x"]
    v = f["restype_ti"].clone()
    v[0, 0, 1, 0] += 1.0                    # no longer one row broadcast
    assert _unbroadcast("restype_ti", v) is v
