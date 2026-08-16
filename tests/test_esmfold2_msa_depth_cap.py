"""The Wormhole MSA depth cap: what it does, and where it must not fire.

The cap exists because an ESMFold2 MSA fold's [B,L,M,c] tensors scale with residues*depth
and a 12 GB Wormhole chip runs out of contiguous space above the product it is measured to
allocate. Two properties are load-bearing and both are pinned here: it is a no-op on
Blackhole and at the measured-good size, and it never raises a requested depth.
"""
import tt_bio.tenstorrent as tt


def _wormhole(monkeypatch):
    monkeypatch.setattr(tt, "is_wormhole", lambda: True)


def test_no_op_off_wormhole(monkeypatch):
    monkeypatch.setattr(tt, "is_wormhole", lambda: False)
    for length in (128, 640, 788, 1024):
        assert tt.msa_depth_cap(length, 8192) == 8192


def test_measured_good_size_is_untouched(monkeypatch):
    """640 aa at depth 8192 is the fold that succeeded, so it must stay byte-identical."""
    _wormhole(monkeypatch)
    assert tt.msa_depth_cap(640, 8192) == 8192


def test_the_failing_band_is_capped_to_the_proven_product(monkeypatch):
    _wormhole(monkeypatch)
    for length in (788, 980, 1024):
        depth = tt.msa_depth_cap(length, 8192)
        assert depth < 8192
        assert length * depth <= tt.WORMHOLE_MSA_AREA


def test_never_deepens_a_shallow_request(monkeypatch):
    _wormhole(monkeypatch)
    assert tt.msa_depth_cap(1024, 256) == 256
    assert tt.msa_depth_cap(128, 32) == 32


def test_degenerate_inputs_pass_through(monkeypatch):
    _wormhole(monkeypatch)
    assert tt.msa_depth_cap(0, 8192) == 8192
    assert tt.msa_depth_cap(1024, 0) == 0
