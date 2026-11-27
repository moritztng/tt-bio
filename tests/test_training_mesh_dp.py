"""CPU unit tests for tt_bio.train.mesh_dp (SPMD MeshDevice DP helpers)."""
import sys
import pytest
import torch

from tt_bio.train import mesh_dp


def test_mesh_dp_import_does_not_import_ttnn():
    import subprocess
    from pathlib import Path
    out = subprocess.run(
        [sys.executable, "-c", "import sys, tt_bio.train.mesh_dp; assert 'ttnn' not in sys.modules"],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.returncode == 0, out.stderr


def test_mesh_world_and_is_multi_chip():
    class FakeDev:
        def __init__(self, n):
            self._n = n
        def get_num_devices(self):
            return self._n

    assert mesh_dp.mesh_world(None) == 1
    assert not mesh_dp.is_multi_chip(None)
    assert mesh_dp.mesh_world(FakeDev(1)) == 1
    assert not mesh_dp.is_multi_chip(FakeDev(1))
    assert mesh_dp.mesh_world(FakeDev(32)) == 32
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
