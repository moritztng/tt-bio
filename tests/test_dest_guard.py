"""dest_guard keeps every fp32-accumulating matmul on Wormhole at K block 1, and touches nothing else.

The rewrites are checked on stand-in tensors and a recording op, so no device is opened: what the
guard hands ttnn is the whole contract. Whether K block 1 is clean on the device is measured by
perf/spd_wherr/guard_check.py.
"""
import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import dest_guard  # noqa: E402


class _Grid:
    x, y = 8, 9


class _Dev:
    def compute_with_storage_grid_size(self):
        return _Grid()


class _T:
    def __init__(self, *shape):
        self.shape = shape

    def is_sharded(self):
        return False

    def device(self):
        return _Dev()


def _ckc(fp32=True, l1acc=False):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                            fp32_dest_acc_en=fp32, packer_l1_acc=l1acc)


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr(dest_guard, "_ON", [True])
    monkeypatch.setattr(dest_guard, "GUARDED", frozenset(dest_guard._CLASSES))
    monkeypatch.setattr(dest_guard, "_CHOSEN", {})
    monkeypatch.setattr(dest_guard, "_REFUSED", {})
    monkeypatch.setattr(dest_guard, "STATS", {"rewritten": 0, "kept": 0, "refused": 0})
    return dest_guard


def _recorder(refuse=lambda kw: False):
    calls = []

    def op(*args, **kw):
        calls.append(kw)
        if refuse(kw):
            raise RuntimeError("TT_FATAL: L1")
        return "y"
    return op, calls


_MM = lambda args, kw: (args[0], args[1])


def test_explicit_config_goes_to_k1_with_l1_acc(guard):
    op, calls = _recorder()
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 9), in0_block_w=4, out_subblock_h=1, out_subblock_w=2, out_block_h=4,
        out_block_w=4, per_core_M=4, per_core_N=4, transpose_mcast=False, fused_activation=None, fuse_batch=True)
    f = guard._wrap("matmul", op, _MM)
    assert f(_T(1024, 768), _T(768, 1024), compute_kernel_config=_ckc(), program_config=pc) == "y"
    got = calls[-1]
    assert got["program_config"].in0_block_w == 1
    assert (got["program_config"].per_core_M, got["program_config"].out_subblock_w) == (4, 2)
    assert got["compute_kernel_config"].packer_l1_acc and got["compute_kernel_config"].fp32_dest_acc_en
    f(_T(1024, 768), _T(768, 1024), compute_kernel_config=_ckc(), program_config=pc)
    assert calls[-1]["program_config"].in0_block_w == 1 and guard.STATS["rewritten"] == 2


def test_minimal_matmul_k_block(guard):
    op, calls = _recorder()
    cfg = ttnn.MinimalMatmulConfig(M_block_size=4, K_block_size=8, N_block_size=1, subblock_h=4, subblock_w=1,
                                   compute_with_storage_grid_size=ttnn.CoreCoord(8, 9))
    f = guard._wrap("minimal_matmul", op, _MM)
    f(_T(8192, 256), _T(256, 256), compute_kernel_config=_ckc(), config=cfg)
    assert calls[-1]["config"].K_block_size == 1 and calls[-1]["config"].M_block_size == 4


def test_unconfigured_call_falls_back_to_a_smaller_block_and_remembers_it(guard):
    op, calls = _recorder(refuse=lambda kw: kw.get("program_config") is not None
                          and kw["program_config"].out_block_h * kw["program_config"].out_block_w > 32)
    f = guard._wrap("linear", op, _MM)
    x, w = _T(5, 2560, 768), _T(768, 3072)
    f(x, w, compute_kernel_config=_ckc(), core_grid=ttnn.CoreGrid(y=9, x=8))
    ok = calls[-1]["program_config"]
    assert ok.in0_block_w == 1 and ok.out_block_h * ok.out_block_w <= 32 and "core_grid" not in calls[-1]
    n = len(calls)
    f(x, w, compute_kernel_config=_ckc(), core_grid=ttnn.CoreGrid(y=9, x=8))
    assert len(calls) == n + 1 and calls[-1]["program_config"].out_block_h == ok.out_block_h


def test_every_rewrite_refused_runs_the_call_as_written(guard):
    op, calls = _recorder(refuse=lambda kw: kw.get("program_config") is not None)
    f = guard._wrap("linear", op, _MM)
    ck = _ckc()
    assert f(_T(2560, 768), _T(768, 768), compute_kernel_config=ck) == "y"
    assert calls[-1] == {"compute_kernel_config": ck} and guard.STATS["refused"] == 1


@pytest.mark.parametrize("ck", [_ckc(fp32=False), None])
def test_bf16_dest_is_untouched(guard, ck):
    op, calls = _recorder()
    f = guard._wrap("linear", op, _MM)
    f(_T(2560, 768), _T(768, 768), compute_kernel_config=ck)
    assert calls[-1] == {"compute_kernel_config": ck} and guard.STATS["rewritten"] == 0


def test_off_device_is_untouched(guard, monkeypatch):
    monkeypatch.setattr(dest_guard, "_ON", [False])
    assert not guard.exposed(_ckc())
    assert guard.descriptor((4, 8, 4, 2, 2), (None, False, True, False)) == (4, 8, 4, 2, 2)


def test_generic_descriptor_and_class_list(guard, monkeypatch):
    assert guard.descriptor((4, 8, 4, 2, 2), (None, False, True, False)) == (4, 1, 4, 2, 2)
    assert guard.descriptor((4, 8, 4, 2, 2), (None, False, False, False)) == (4, 8, 4, 2, 2)
    monkeypatch.setattr(dest_guard, "GUARDED", frozenset({"pc"}))
    assert guard.descriptor((4, 8, 4, 2, 2), (None, False, True, False)) == (4, 8, 4, 2, 2)
    assert not guard.tail_exposed(_ckc())
    op, calls = _recorder()
    guard._wrap("linear", op, _MM)(_T(2560, 768), _T(768, 768), compute_kernel_config=_ckc())
    assert "program_config" not in calls[-1]
