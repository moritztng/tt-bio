"""Resident device modules (tenstorrent.resident_weights): reused only for the same checkpoint, place, levers and device."""
from tt_bio import tenstorrent as T


class _Wrapper:
    pass


def _load(prefix="trunk.", wrapper=_Wrapper):
    return T._resident_module(prefix, wrapper, object)


def test_reused_only_inside_an_opt_in(monkeypatch):
    monkeypatch.setattr(T, "_RESIDENT", {})
    assert _load() is not _load()  # no tag: every load builds
    with T.resident_weights(("a.ckpt",)):
        first = _load()
        assert _load() is first
        assert _load("diffusion.") is not first
    with T.resident_weights(("b.ckpt",)):
        assert _load() is not first
    with T.resident_weights(("a.ckpt",)):
        assert _load() is first


def test_a_lever_change_or_device_reopen_rebuilds_and_frees(monkeypatch):
    monkeypatch.setattr(T, "_RESIDENT", {})
    with T.resident_weights(("a.ckpt",)):
        first = _load()
        with T.levers(["conf_bins"]):
            assert _load() is not first
        assert len(T._RESIDENT) == 1  # one lever set kept at a time
        normal = _load()
        monkeypatch.setattr(T, "_device_generation", T._device_generation + 1)
        assert _load() is not normal
        assert len(T._RESIDENT) == 1


def test_a_predict_load_frees_them(monkeypatch):
    from tt_bio import worker

    monkeypatch.setattr(T, "_RESIDENT", {})
    with T.resident_weights(("a.ckpt",)):
        _load()
    worker._free_design_engines()
    assert T._RESIDENT == {}
