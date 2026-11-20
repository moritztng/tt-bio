"""The template embedder runs its pairformer stack once per DISTINCT template.

Without real templates, protenix_data.dummy_template_features emits one gap slot and three
identical ALA slots, so the trunk runs 2 stacks per recycle instead of 4. The mean still adds one
term per slot in slot order, so the arithmetic is the same adds the 4-stack path did. CPU only.
"""
import torch

from tt_bio.protenix import unique_slots
from tt_bio.protenix_data import dummy_template_features


def test_unique_slots_exact_and_ordered():
    a, b = torch.zeros(3, 4), torch.ones(3, 4)
    c = b.clone(); c[0, 0] = 1.0000001
    d, s = unique_slots([a, b, b.clone(), c, a.clone()])
    assert s == [0, 1, 1, 2, 0]
    assert len(d) == 3 and d[0] is a and d[1] is b and d[2] is c


def test_dummy_templates_collapse_to_two_stacks():
    f = dummy_template_features(37)
    import torch.nn.functional as F
    feats = [torch.cat([F.one_hot(f["template_aatype"][t], 32).float(), f["template_distogram"][t].sum(-1, keepdim=True)], -1)
             for t in range(f["template_aatype"].shape[0])]
    d, s = unique_slots(feats)
    assert s == [0, 1, 1, 1] and len(d) == 2
