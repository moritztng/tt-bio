"""protenix-v1/v2 replay their diffusion step from a trace by default on Blackhole only.

--trace / --no_trace decide when given; unset (None) is the shipped default, read off the arch.
A fold replays only when the chip it got carries the Protenix region, because a resident worker
does not reload on --trace. No device is opened here.
"""
import pytest

import tt_bio.tenstorrent as T
from tt_bio import worker as W


@pytest.mark.parametrize("arch, want", [("blackhole", True), ("wormhole_b0", False)])
def test_unset_follows_the_arch(monkeypatch, arch, want):
    monkeypatch.setattr(T, "arch_name", lambda: arch)
    assert W._protenix_trace({"model": "protenix-v2"}) is want
    assert W._protenix_trace({"model": "protenix-v1", "trace": None}) is want


@pytest.mark.parametrize("arch", ["blackhole", "wormhole_b0"])
def test_an_explicit_choice_wins(monkeypatch, arch):
    monkeypatch.setattr(T, "arch_name", lambda: arch)
    assert W._protenix_trace({"trace": True}) is True
    assert W._protenix_trace({"trace": False}) is False


def test_a_fold_replays_only_on_a_region_that_fits(monkeypatch):
    monkeypatch.setattr(T, "trace_region_bytes", lambda capture: 36 << 20)
    for size, want in [(0, False), (27 << 20, False), (36 << 20, True)]:
        monkeypatch.setattr(T, "trace_region_size", lambda s=size: s)
        assert W._protenix_region_reserved() is want


def test_the_cli_leaves_trace_unset_by_default():
    from tt_bio.main import predict

    opt = next(p for p in predict.params if p.name == "trace")
    assert opt.default is None and "--no_trace" in opt.secondary_opts
