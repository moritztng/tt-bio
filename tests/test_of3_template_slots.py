"""derived_template_slots dedups the raw template keys and matches dedup-after-derive exactly."""
import torch

from tt_bio.openfold3_host_prep import (dedup_template_slots, derive_template_feat,
                                        derived_template_slots)


def _raw(n=9, nt=4):
    g = torch.Generator().manual_seed(0)
    one = {
        "template_pseudo_beta_mask": (torch.rand(1, n, generator=g) > 0.2).float(),
        "template_backbone_frame_mask": (torch.rand(1, n, generator=g) > 0.2).float(),
        "template_restype": torch.randint(0, 32, (1, n, 32), generator=g, dtype=torch.int32),
        "template_unit_vector": torch.randn(1, n, n, 3, generator=g),
        "template_distogram": torch.rand(1, n, n, 39, generator=g),
    }
    f = {k: v.expand(nt, *v.shape[1:]).clone() for k, v in one.items()}
    f["asym_id"] = torch.tensor([0] * 5 + [1] * (n - 5))
    return f


def _check(f):
    want_feat, want_slots = dedup_template_slots(derive_template_feat(f))
    feat, slots = derived_template_slots(f)
    assert slots == want_slots and feat.keys() == want_feat.keys()
    for k in feat:
        assert feat[k].dtype == want_feat[k].dtype and torch.equal(feat[k], want_feat[k]), k
    return slots


def test_identical_slots_collapse_to_one():
    assert _check(_raw()) == [0, 0, 0, 0]


def test_a_slot_differing_in_any_raw_key_is_kept():
    for key, idx in (("template_distogram", (2, 3, 4, 5)), ("template_pseudo_beta_mask", (1, 6)),
                     ("template_backbone_frame_mask", (3, 0)), ("template_restype", (2, 1, 7)),
                     ("template_unit_vector", (1, 8, 2, 0))):
        f = _raw()
        f[key][idx] = f[key][idx] + 1
        slot = idx[0]
        assert _check(f) == [1 if t == slot else 0 for t in range(4)], key
