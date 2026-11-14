# Host-only: how many launches gin_moved takes ESMFold2's c_z 256 in-projection in, with and without the fused gate.
import types

import pytest

from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL

BH_BANK = 1461760   # the Blackhole L1 allocator's bank (tenstorrent._l1_bank_bytes)


@pytest.mark.parametrize("bank,fuse,want_n,want_out", [
    (BH_BANK, False, 2, 64),     # 644 tiles: two launches, the pair tensor read twice
    (BH_BANK, True, 1, 64),      # 580 tiles: one launch, output still double-buffered
    (1350000, True, 1, 32),      # a smaller bank: one launch on a single-buffered output
    (1000000, True, 4, 64),      # too small for one launch either way
])
def test_esmfold2_gin_launches(monkeypatch, bank, fuse, want_n, want_out):
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: bank)
    monkeypatch.setattr(TTL, "_lever", lambda name: fuse and name == "trimul_gin_fuse")
    x = types.SimpleNamespace(shape=(1, 512, 512, 256))
    assert TTL.gin_move_split(x, 256) == want_n
    if want_n == 1:
        assert TTL._gin_out_tiles(8, 16, False) == want_out


def test_fused_gate_never_takes_a_masked_call(monkeypatch):
    monkeypatch.setattr(TTL, "_lever", lambda name: name == "trimul_gin_fuse")
    assert TTL._gin_fuse(False) and not TTL._gin_fuse(True)
    monkeypatch.setattr(TTL, "SIGPOLY", False)   # metal's sigmoid has no in-DST gate form
    assert not TTL._gin_fuse(False)
