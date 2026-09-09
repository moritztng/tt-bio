"""A manifest describes the arrays a run actually wrote.

`tt-bio saprot` used to patch the written JSON after the fact, unconditionally, so a run
without `--logits` reported `"logits": false` next to a `shapes.logits` describing a
[length, 446] array that was not in the npz. The shape is now an argument to the one
manifest writer, so the flag and the shape cannot disagree.
"""
import json

import pytest

from tt_bio.esmc import LOGITS_SHAPE, SAPROT_LOGITS_SHAPE, write_manifest_for


def _manifest(tmp_path, **kw):
    p = tmp_path / "manifest.json"
    write_manifest_for([("A", 10), ("B", 20)], 1152, p, model="esmc-600m", pool="mean",
                       fast=False, out_format="npz", **kw)
    return json.loads(p.read_text())


def test_no_logits_means_no_logits_shape(tmp_path):
    m = _manifest(tmp_path, return_logits=False)
    assert m["logits"] is False
    assert m["shapes"]["logits"] is None


def test_no_logits_shape_even_when_a_shape_is_offered(tmp_path):
    m = _manifest(tmp_path, return_logits=False, logits_shape=SAPROT_LOGITS_SHAPE)
    assert m["logits"] is False
    assert m["shapes"]["logits"] is None


@pytest.mark.parametrize("shape,needle", [(LOGITS_SHAPE, "64"), (SAPROT_LOGITS_SHAPE, "446")])
def test_the_written_shape_is_the_one_asked_for(tmp_path, shape, needle):
    m = _manifest(tmp_path, return_logits=True, logits_shape=shape)
    assert m["logits"] is True
    assert needle in m["shapes"]["logits"]


def test_manifest_lists_every_sequence(tmp_path):
    m = _manifest(tmp_path, return_logits=False)
    assert [e["id"] for e in m["sequences"]] == ["A", "B"]
    assert [e["file"] for e in m["sequences"]] == ["A.npz", "B.npz"]
