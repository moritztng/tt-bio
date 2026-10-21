"""`protenix.template_sources`: which template slots `Trunk._template` computes once and re-adds.

Equal slots give bit-identical embedder outputs, so the trunk runs the template pairformer once per distinct slot.
A fold without real templates carries 4 dummy slots of which 1-3 are equal, so it computes 2."""
import torch

from tt_bio.protenix import template_sources
from tt_bio.protenix_data import dummy_template_features


def test_dummy_slots_one_to_three_share_slot_one():
    assert template_sources(dummy_template_features(37)) == [0, 1, 1, 1]


def test_distinct_slots_each_compute():
    f = dummy_template_features(37)
    f["template_aatype"][2, 5] = 7
    f["template_distogram"][3, 1, 2, 0] = 1.0
    assert template_sources(f) == [0, 1, 2, 3]


def test_equal_to_an_earlier_distinct_slot():
    f = dummy_template_features(37)
    f["template_aatype"][2] = f["template_aatype"][0]
    assert template_sources(f) == [0, 1, 0, 1]


def test_geometry_alone_separates_slots():
    f = dummy_template_features(37)
    f["template_unit_vector"][3, 4, 4] = torch.tensor([1.0, 0.0, 0.0])
    assert template_sources(f) == [0, 1, 1, 3]


def test_no_template_features():
    assert template_sources({}) == []
