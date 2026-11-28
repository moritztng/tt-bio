"""The card convolution's own code on the host: ``ttnn`` replaced by a float64 torch stand-in,
so the block cuts, gathers and scatters of :class:`tt_bio.diffdock.device.DeviceConvLayer` are
checked against the host layer without a card. What a card adds is precision, not structure.
"""

import sys
import types

import pytest
import torch

from tt_bio.diffdock.conv import TensorProductConvLayer, blocks, bucket

IN, SH, OUT, NEF = "16x0e + 4x1o", "1x0e + 1x1o + 1x2e", "16x0e + 4x1o + 4x1e + 4x0o", 24


@pytest.fixture
def host_ttnn(monkeypatch):
    f64, F = torch.float64, torch.nn.functional
    t = types.ModuleType("ttnn")
    t.Tensor, t.float32, t.bfloat16, t.uint32 = torch.Tensor, "f32", "bf16", "u32"
    t.TILE_LAYOUT, t.ROW_MAJOR_LAYOUT = "tile", "rm"
    t.MathFidelity = types.SimpleNamespace(HiFi4=4)
    t.Arch = types.SimpleNamespace(BLACKHOLE="bh")
    t.types = types.SimpleNamespace(BlackholeComputeKernelConfig=lambda **kw: kw)
    t.from_torch = lambda x, dtype=None, **kw: x.long() if dtype == "u32" else x.to(f64)
    t.to_torch = lambda x: x
    t.matmul = lambda a, b, **kw: a @ b
    t.linear = lambda a, w, bias=None, **kw: a @ w + bias
    t.embedding = lambda idx, table, **kw: table[idx]
    t.to_layout = t.typecast = lambda x, _: x
    t.reshape = lambda x, shape: x.reshape(shape)
    t.concat = lambda parts, dim=0: torch.cat(parts, dim)
    t.pad = lambda x, pads, value=0.0: F.pad(x, (0, pads[1][1], 0, pads[0][1]), value=value)
    t.clamp = lambda x, min=None: x.clamp(min=min)
    t.deallocate = lambda x: None
    for name in ("relu", "add", "subtract", "multiply", "reciprocal"):
        setattr(t, name, getattr(torch, name))
    monkeypatch.setitem(sys.modules, "ttnn", t)
    return types.SimpleNamespace(arch=lambda: "bh")


@pytest.mark.parametrize("groups", [1, 2])
def test_card_layer_in_row_blocks_matches_the_host_layer(host_ttnn, groups):
    from tt_bio.diffdock.device import DeviceConvModule
    torch.manual_seed(0)
    layer = TensorProductConvLayer(IN, SH, OUT, NEF, edge_groups=groups).eval()
    N, sizes = 150, [1500, 700][:groups]
    E = sum(sizes)
    assert len(blocks(bucket(N))) > 1 and all(len(blocks(bucket(n))) > 1 for n in sizes)
    mod = DeviceConvModule(host_ttnn, layer, grade=True)
    x = torch.randn(N, layer.in_irreps.dim, dtype=torch.float64)
    ei = torch.randint(0, N, (2, E))
    ea = list(torch.randn(E, NEF, dtype=torch.float64).split(sizes))
    sh = torch.randn(E, 9, dtype=torch.float64)
    mask = (torch.rand(E) > 0.2).double()
    mod(x, ei, ea if groups > 1 else ea[0], sh, edge_mask=mask)
    assert mod.log[0]["rel_frobenius"] < 1e-5, mod.log[0]


def test_card_stack_in_row_blocks_matches_the_host_stack(host_ttnn):
    """The resident form: edge input built on the card from node rows, two FC groups."""
    from tt_bio.diffdock import static_net
    from tt_bio.diffdock.device import DeviceConvModule
    torch.manual_seed(1)
    ns, a, N, sizes = 16, 8, 150, [1500, 700]
    E = sum(sizes)
    layers = [TensorProductConvLayer(i, SH, OUT, a + 2 * ns, edge_groups=2).double().eval()
              for i in (IN, OUT)]
    x = torch.randn(N, layers[0].in_irreps.dim, dtype=torch.float64)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, a, dtype=torch.float64)
    sh = torch.randn(E, 9, dtype=torch.float64)
    mask = (torch.rand(E) > 0.2).double()
    plan = [(E, (sizes[0],))] * 2
    with torch.no_grad():
        ref = static_net.run_stack(layers, x, ei, ea, sh, mask, ns, plan)
        got = static_net.run_stack([DeviceConvModule(host_ttnn, lyr) for lyr in layers],
                                   x, ei, ea, sh, mask, ns, plan)
    assert (got - ref).norm() / ref.norm() < 1e-5
