"""drop_weight_caches empties every `_wc*` cache under a module and leaves the rest alone."""
import pytest

pytest.importorskip("ttnn")
from tt_bio.protenix import drop_weight_caches  # noqa: E402


class Mod:  # a tt_bio class, so the walk enters it
    pass


Mod.__module__ = "tt_bio.fake"


class Foreign:
    pass


def test_drop_weight_caches_walks_tt_bio_objects_only():
    inner, outer, other = Mod(), Mod(), Foreign()
    inner._wc_dit = {("w", True): 1}
    other._wc = {"k": 2}
    outer._wc, outer._wc16, outer._adaln_cache = {"a": 1}, {"b": 1}, {"p": 3}
    outer.blocks, outer.sub, outer.foreign = [inner], inner, other
    outer.self_ref = outer
    drop_weight_caches(outer)
    assert outer._wc == {} and outer._wc16 == {} and inner._wc_dit == {}
    assert outer._adaln_cache == {"p": 3}   # not a lazily uploaded weight cache
    assert other._wc == {"k": 2}            # not a tt_bio object
