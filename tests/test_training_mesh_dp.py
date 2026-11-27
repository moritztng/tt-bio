"""CPU unit tests for tt_bio.train.mesh_dp (SPMD MeshDevice DP helpers)."""
import sys
import pytest
import torch

import importlib.util
from pathlib import Path
from tt_bio.train import mesh_dp

_ccl_spec = importlib.util.spec_from_file_location("_test_training_ccl", Path(__file__).with_name("test_training_ccl.py"))
_ccl_mod = importlib.util.module_from_spec(_ccl_spec)
_ccl_spec.loader.exec_module(_ccl_mod)
_FakeMeshDevice, _make_mock_ttnn = _ccl_mod._FakeMeshDevice, _ccl_mod._make_mock_ttnn


def test_mesh_dp_import_does_not_import_ttnn():
    assert "ttnn" not in sys.modules


def test_mesh_world_and_is_multi_chip():
    class FakeDev:
        def __init__(self, n, shape=None):
            self._n = n
            if shape is not None:
                self.shape = shape

        def get_num_devices(self):
            return self._n

    assert mesh_dp.mesh_world(None) == 1
    assert mesh_dp.mesh_shape(None) == (1, 1)
    assert not mesh_dp.is_multi_chip(None)
    assert mesh_dp.mesh_world(FakeDev(1)) == 1
    assert not mesh_dp.is_multi_chip(FakeDev(1))
    assert mesh_dp.mesh_world(FakeDev(32, (8, 4))) == 32
    assert mesh_dp.mesh_shape(FakeDev(32, (8, 4))) == (8, 4)
    assert mesh_dp.is_multi_chip(FakeDev(32))


def test_stack_batches_and_unstack_leading_roundtrip():
    world, k, n = 4, 2, 16
    batches = []
    for r in range(world):
        g = torch.Generator().manual_seed(100 + r)
        batches.append({
            "padding_mask": torch.ones(k, n),
            "x_t": {"bb_ca": torch.randn(k, n, 3, generator=g)},
            "tag": "const_meta",
        })
    stacked = mesh_dp.stack_batches(batches)
    assert stacked["padding_mask"].shape == (world * k, n)
    assert stacked["x_t"]["bb_ca"].shape == (world * k, n, 3)
    assert stacked["tag"] == "const_meta"

    parts = mesh_dp.unstack_leading(stacked["x_t"]["bb_ca"], world)
    assert len(parts) == world
    for r in range(world):
        assert torch.equal(parts[r], batches[r]["x_t"]["bb_ca"])

    with pytest.raises(ValueError):
        mesh_dp.unstack_leading(torch.zeros(5, 3), 4)


def test_batched_host_glue_backward_equals_rank_summed_serial_gradient():
    """Verify stacked [world, ...] PyTorch backward equals sum_r serial backward (dp equiv)."""
    world, n, d_in, d_out = 8, 12, 16, 8
    g = torch.Generator().manual_seed(99)
    lin = torch.nn.Linear(d_in, d_out, bias=True)
    xs = [torch.randn(1, n, d_in, generator=g) for _ in range(world)]
    seeds = [torch.randn(1, n, d_out, generator=g) for _ in range(world)]

    # Serial accumulation across ranks
    lin.zero_grad()
    for x, sd in zip(xs, seeds):
        y = lin(x)
        torch.autograd.backward([y], [sd])
    w_ser = lin.weight.grad.clone()
    b_ser = lin.bias.grad.clone()

    # Single batched SPMD MeshBridge-style backward
    lin.zero_grad()
    y_all = lin(torch.cat(xs, dim=0))
    torch.autograd.backward([y_all], [torch.cat(seeds, dim=0)])
    assert torch.allclose(lin.weight.grad, w_ser, rtol=1e-5, atol=1e-6)
    assert torch.allclose(lin.bias.grad, b_ser, rtol=1e-5, atol=1e-6)

    # Broken arm: dropping rank world-1 must fail clearly
    lin.zero_grad()
    seeds_bad = list(seeds)
    seeds_bad[-1] = torch.zeros_like(seeds[-1])
    y_bad = lin(torch.cat(xs, dim=0))
    torch.autograd.backward([y_bad], [torch.cat(seeds_bad, dim=0)])
    assert float((lin.weight.grad - w_ser).abs().max()) > 1e-1


def test_mock_mesh_helpers_consensus_axis0_sum_and_bridge_up_replicated(monkeypatch):
    """Verify put_sharded, put_sharded_2d, read_axis0_sum, consensus, and MeshBridge on mock MeshDevice."""
    mock_ttnn = _make_mock_ttnn()
    mock_ttnn.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "ttnn", mock_ttnn)
    rm, cm = 4, 4
    world = rm * cm
    dev = _FakeMeshDevice(rm, cm)

    # 1. put_sharded + read_sharded roundtrip
    x = torch.arange(world * 2 * 8, dtype=torch.float32).reshape(world * 2, 8)
    tx = mesh_dp.put_sharded(x, dev)
    assert tx.shape == (2, 8)
    assert torch.equal(mesh_dp.read_sharded(tx, dev), x)
    tx.deallocate()

    # 2. put_sharded_2d + read_axis0_sum (reads only rm row reps!)
    row_vals = torch.tensor([1.5, 2.25, 3.75, 4.5], dtype=torch.float32).reshape(rm, 1, 1)
    tr = mesh_dp.put_sharded_2d(row_vals, dev, (rm, cm), dims=(0, None))
    assert abs(mesh_dp.read_axis0_sum(tr, (rm, cm), dev) - 12.0) < 1e-12
    tr.deallocate()

    # 3. assert_mesh_consensus: PASS when identical, FAIL when one rank differs
    rep = mock_ttnn.from_torch(torch.ones(4, 4, dtype=torch.float32) * 3.25, device=dev)
    dig = mesh_dp.assert_mesh_consensus(rep, dev, name="ok_param")
    assert len(dig) == 16
    rep._buf.shards[7] = rep._buf.shards[7] + 1e-4
    with pytest.raises(AssertionError, match="rank 7 differs"):
        mesh_dp.assert_mesh_consensus(rep, dev, name="bad_param")
    rep.deallocate()

    # 4. MeshBridge.up (sharded activation) + MeshBridge.up_replicated (replicated model weight)
    import types as _types

    class _FakeAgTensor:
        def __init__(self, value, requires_grad=False):
            self.value = value
            self.requires_grad = requires_grad
            self.grad = None

    monkeypatch.setitem(sys.modules, "tt_bio.autograd", _types.SimpleNamespace(Tensor=_FakeAgTensor))
    g = torch.Generator().manual_seed(321)
    lin = torch.nn.Linear(6, 4, bias=True)
    w_rep = torch.nn.Parameter(torch.randn(4, 6, generator=g))
    xs = [torch.randn(1, 5, 6, generator=g) for _ in range(world)]
    act_grads = [torch.randn(1, 5, 4, generator=g) for _ in range(world)]
    w_grads = [torch.randn(4, 6, generator=g) for _ in range(world)]

    # Serial reference
    lin.zero_grad()
    w_rep.grad = None
    for x_i, ag_i, wg_i in zip(xs, act_grads, w_grads):
        y_i = lin(x_i)
        torch.autograd.backward([y_i, w_rep], [ag_i, wg_i])
    ref_lin_w = lin.weight.grad.clone()
    ref_w_rep = w_rep.grad.clone()

    # MeshBridge batched + replicated backward
    lin.zero_grad()
    w_rep.grad = None
    br = mesh_dp.MeshBridge(dev, dtype=mock_ttnn.float32)
    y_all = lin(torch.cat(xs, dim=0))
    d_act = br.up(y_all, dtype=mock_ttnn.float32)
    d_wrep = br.up_replicated(w_rep, dtype=mock_ttnn.float32)
    d_act.grad = dev.make_tensor([ag_i for ag_i in act_grads])
    d_wrep.grad = dev.make_tensor([wg_i for wg_i in w_grads])
    br.backward_ups()

    assert torch.allclose(lin.weight.grad, ref_lin_w, rtol=1e-5, atol=1e-6)
    assert torch.allclose(w_rep.grad, ref_w_rep, rtol=1e-5, atol=1e-6)
    d_act.value.deallocate()
    d_wrep.value.deallocate()
    assert len(dev.live_bufs) == 0
