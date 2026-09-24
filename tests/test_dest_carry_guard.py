"""The dest-carry guard over ttnn.matmul / ttnn.linear, without a device.

On Wormhole with HiFi4 and fp32 dest a K block wider than one tile returns rare -2^k elements
(`tenstorrent.dest_carry_fault`), so the guard hands every call that leaves the plan to ttnn a
K-block-1 plan. Everywhere else it must be the call the caller wrote, argument for argument:
that is what "Blackhole unchanged" means, and it is checked here by construction rather than by
a digest, because a digest can agree by luck and an identity cannot.
"""
from types import SimpleNamespace

import pytest

ttnn = pytest.importorskip("ttnn")
T = pytest.importorskip("tt_bio.tenstorrent")


class _Mem:
    def is_sharded(self):
        return False


def _tensor(*shape):
    return SimpleNamespace(layout=ttnn.TILE_LAYOUT, padded_shape=list(shape), dtype=ttnn.bfloat16,
                           memory_config=lambda: _Mem())


def _ckc(fid=ttnn.MathFidelity.HiFi4, dest=True):
    return ttnn.types.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                                  fp32_dest_acc_en=dest, packer_l1_acc=True)


@pytest.fixture
def seen(monkeypatch):
    calls = []
    guarded = T._dest_carry_guard(lambda a, b, *args, **kw: calls.append((a, b, args, kw)) or "out")
    return guarded, calls


def _set_arch(monkeypatch, wormhole):
    monkeypatch.setattr(T, "_wormhole", lambda: wormhole)
    monkeypatch.setattr(T, "_DEST_CARRY_GUARD", True)


def test_blackhole_call_is_untouched(seen, monkeypatch):
    _set_arch(monkeypatch, False)
    guarded, calls = seen
    a, b, kw = _tensor(1, 1, 1024, 768), _tensor(768, 1536), dict(compute_kernel_config=_ckc(), core_grid=7)
    assert guarded(a, b, **kw) == "out"
    assert calls == [(a, b, (), kw)]


@pytest.mark.parametrize("ckc", [_ckc(ttnn.MathFidelity.HiFi3), _ckc(dest=False), None])
def test_other_compute_configs_are_untouched(seen, monkeypatch, ckc):
    _set_arch(monkeypatch, True)
    guarded, calls = seen
    a, b, kw = _tensor(1, 1, 1024, 768), _tensor(768, 1536), dict(compute_kernel_config=ckc)
    guarded(a, b, **kw)
    assert calls == [(a, b, (), kw)]


def test_callers_own_plan_is_untouched(seen, monkeypatch):
    _set_arch(monkeypatch, True)
    guarded, calls = seen
    a, b, kw = _tensor(1, 1, 1024, 768), _tensor(768, 1536), dict(compute_kernel_config=_ckc(), program_config="p")
    guarded(a, b, **kw)
    assert calls == [(a, b, (), kw)]


@pytest.mark.parametrize("ta,tb,a_shape,b_shape,m_tiles,n_tiles", [
    (False, False, (1, 1, 1024, 768), (768, 1536), 32, 48),
    (False, True, (1, 1, 1024, 768), (1536, 768), 32, 48),
    (True, False, (1, 1, 768, 1024), (768, 1536), 32, 48),
    (False, False, (4, 256, 768), (768, 1536), 32, 48),
])
def test_wormhole_hifi4_gets_k_block_1(seen, monkeypatch, ta, tb, a_shape, b_shape, m_tiles, n_tiles):
    _set_arch(monkeypatch, True)
    plans = []
    monkeypatch.setattr(T, "_k1_program_config",
                        lambda m, n, fused_activation=None, out_bytes=2: plans.append((m, n)) or "k1")
    guarded, calls = seen
    guarded(_tensor(*a_shape), _tensor(*b_shape), compute_kernel_config=_ckc(), core_grid=7,
            transpose_a=ta, transpose_b=tb)
    kw = calls[0][3]
    assert plans == [(m_tiles, n_tiles)]
    assert kw["program_config"] == "k1" and "core_grid" not in kw
    assert (kw["transpose_a"], kw["transpose_b"]) == (ta, tb)


@pytest.mark.parametrize("a_shape,b_shape", [((1, 1, 1024, 32), (32, 1536)),     # one K tile
                                             ((1, 8, 1024, 768), (1, 8, 768, 64))])  # batched b
def test_unplannable_calls_run_as_written(seen, monkeypatch, a_shape, b_shape):
    _set_arch(monkeypatch, True)
    guarded, calls = seen
    a, b, kw = _tensor(*a_shape), _tensor(*b_shape), dict(compute_kernel_config=_ckc())
    guarded(a, b, **kw)
    assert calls == [(a, b, (), kw)]


@pytest.mark.device
def test_k1_plan_fits_the_bank():
    cfg = T._k1_program_config(512, 96)
    assert cfg.in0_block_w == 1
    assert cfg.out_subblock_h * cfg.out_subblock_w <= 4
    assert cfg.per_core_M % cfg.out_block_h == 0 and cfg.per_core_N % cfg.out_block_w == 0
    assert T._matmul_cb_bytes(1, cfg.out_block_h, cfg.out_block_w, 2) <= T._matmul_cb_budget()
