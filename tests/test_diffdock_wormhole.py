"""DiffDock-L on Wormhole: the Blackhole device program, the same golden, the same bar.

There is one device path (`tt_bio/diffdock/device.py`) for both architectures. A Wormhole
card runs the convolutions the Blackhole grade ran (`scripts/diffdock_port/device_conv_grade.py`,
`state/dfd/DEVICE.md` section 3) and is graded the same way: against the host layer in
float64, at the released checkpoint's own layer signatures, with pad edges, a random mask and
a node whose every edge is masked. The bar is one bar. A Wormhole result that needs a looser
one is a value difference, and a result that depends on the card that ran it is a bug, not a
per-arch tolerance.

The card-free half below proves the golden is the device program: the device rewrite run in
float64 on the host matches the host tensor product to rounding, so a card that clears the bar
is running the right arithmetic, not a second approximation of it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import device as ddev  # noqa: E402
from tt_bio.diffdock.conv import TensorProductConvLayer  # noqa: E402
from tt_bio.diffdock.harmonics import spherical_harmonics  # noqa: E402

SH = "1x0e + 1x1o + 1x2e"
#: The torsion head's "harmonics": the full tensor product of SH with the 2e bond direction.
TOR_SH = "1x0e + 1x1o + 1x1e + 1x2o + 1x2e + 1x2e + 1x3o + 1x3e + 1x4e"
FULL = "48x0e + 10x1o + 10x1e + 10x0o"
#: The released score network's layer signatures (`DiffDockConfig.layer_plan`), the three the
#: Blackhole conv grade ran plus the two heads:
#: (in, sh, out, edge groups, edge features = hidden, residual).
LAYERS = {
    "rec_emb_layers.0": ("48x0e", SH, "48x0e + 10x1o", 1, 144, True),
    "conv_layers.0": (FULL, SH, FULL, 4, 144, True),
    "conv_layers.10": (FULL, SH, FULL, 2, 144, True),
    "final_conv": (FULL, SH, "2x1o + 2x1e", 1, 96, False),
    "tor_bond_conv": (FULL, TOR_SH, "48x0o + 48x0e", 1, 144, False),
}

#: Blackhole's grade at 1350 MHz, real v1.1 weights, worst of three layers: PCC 0.9999985,
#: max rel err 4.3e-3, rel Frobenius 1.8e-3 (state/dfd/DEVICE.md section 3). The bar leaves
#: that envelope room and holds both architectures to it.
BAR = dict(pcc_min=0.99999, max_rel_err=1e-2, rel_frobenius=5e-3)


def _checkpoint():
    env = os.environ.get("DIFFDOCK_MODELS")
    for p in (Path(env) if env else None, Path.home() / "scratch/dfd/ckpt/extracted"):
        if p is not None and (p / "score_model/best_ema_inference_epoch_model.pt").is_file():
            return p / "score_model/best_ema_inference_epoch_model.pt"
    return None


def _layer(name: str) -> TensorProductConvLayer:
    """The layer at its released signature: the checkpoint's weights when present, else seeded.

    Seeded weights have the same shapes and torch's default init; the bar is relative, so
    either grades the arithmetic. The checkpoint is preferred because it is what users run.
    """
    a, sh, b, groups, nef, residual = LAYERS[name]
    torch.manual_seed(0)
    m = TensorProductConvLayer(a, sh, b, nef, hidden_features=nef, edge_groups=groups,
                               residual=residual)
    ckpt = _checkpoint()
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        pre = name + "."
        missing, _ = m.load_state_dict(
            {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
        assert not missing, f"{name}: checkpoint lacks {missing}"
    return m.eval().double()


def _case(layer, n_nodes=128, group_edges=1024, seed=1):
    """A random graph as graph.py builds them: tile-sized groups, pad edges onto a pad node,
    a 0.7 mask, and node 0 with eight edges that are all masked."""
    gen = torch.Generator().manual_seed(seed)
    G, N = layer.edge_groups, n_nodes
    E = G * group_edges
    real, pad_node = E - 64, N - 1
    src = torch.randint(1, N - 1, (E,), generator=gen)
    dst = torch.randint(0, N - 1, (E,), generator=gen)
    src[:8] = 0
    src[real:], dst[real:] = pad_node, pad_node
    mask = (torch.rand(E, generator=gen) < 0.7).double()
    mask[:8], mask[real:] = 0.0, 0.0
    x = torch.randn(N, layer.in_irreps.dim, generator=gen, dtype=torch.float64)
    x[pad_node] = 0
    if str(layer.sh_irreps) == SH:
        sh = spherical_harmonics(torch.randn(E, 3, generator=gen, dtype=torch.float64),
                                 (0, 1, 2))
    else:   # the torsion head's product harmonics: any unit-scale block grades the arithmetic
        sh = torch.randn(E, layer.sh_irreps.dim, generator=gen, dtype=torch.float64) / 3.0
    nef = (layer.fc[0] if G > 1 else layer.fc)[0].weight.shape[1]
    groups = [torch.randn(group_edges, nef, generator=gen, dtype=torch.float64)
              for _ in range(G)]
    return torch.stack([src, dst]), x, sh, groups, mask


def _grade(ref, got):
    ref, got = ref.double().flatten(), got.double().flatten()
    return dict(pcc=torch.corrcoef(torch.stack([ref, got]))[0, 1].item(),
                max_rel_err=((got - ref).abs().max() / ref.abs().max()).item(),
                rel_frobenius=(torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item(),
                finite=bool(torch.isfinite(got).all()))


# ------------------------------------------------------------------------------- card-free

@pytest.mark.parametrize("name", sorted(LAYERS))
def test_the_device_program_is_the_golden_in_float64(name):
    """The rewrite the card runs, in float64 on the host, is the host tensor product."""
    layer = _layer(name)
    ei, x, sh, groups, _ = _case(layer, n_nodes=64, group_edges=256)
    fcs = list(layer.fc) if layer.edge_groups > 1 else [layer.fc]
    lo = 0
    for fc, ea in zip(fcs, groups):
        n = ea.shape[0]
        x1, x2 = x[ei[1, lo:lo + n]], sh[lo:lo + n]
        want = layer.tp(x1, x2, fc(ea))
        got = ddev.host_emulation(layer.tp, fc, ea, x1, x2)
        assert (got - want).abs().max() <= 1e-12 * want.abs().max().clamp_min(1.0), name
        lo += n


# ------------------------------------------------------------------------- the Wormhole card

@pytest.mark.device
def test_wh_matches_bh_within_bar(record_property):
    """Every graded layer on a Wormhole card clears Blackhole's bar against float64.

    The all-masked node must come out as its residual (never NaN), and the AICLK sampled
    during the run is recorded beside the grade.
    """
    ttnn = pytest.importorskip("ttnn")
    from tt_bio.tenstorrent import arch_name
    if arch_name() != "wormhole_b0":
        pytest.skip(f"this host's cards are {arch_name()}, not Wormhole; Blackhole's grade is "
                    "scripts/diffdock_port/device_conv_grade.py")
    from tt_bio.diffdock.capture import clock_summary, sample_aiclk
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_aiclk, daemon=True,
                          args=(stop, samples, os.environ.get("TT_VISIBLE_DEVICES", "0")))
    th.start()
    rows, failed = {}, []
    try:
        for name in LAYERS:
            layer = _layer(name)
            ei, x, sh, groups, mask = _case(layer)
            N, E = x.shape[0], ei.shape[1]
            with torch.no_grad():
                ref = layer(x, ei, groups, sh, edge_mask=mask)
            dl = ddev.DeviceConvLayer(dev, layer.float())
            layer.double()
            table = ddev.to_device(dev, x)
            got = dl(table, dl.graph(ei, N, N), [ddev.to_device(dev, t) for t in groups],
                     ddev.to_device(dev, sh),
                     ttnn.from_torch(mask.float().reshape(E, 1), dtype=ttnn.float32,
                                     layout=ttnn.TILE_LAYOUT, device=dev))
            got = ttnn.to_torch(got)[:, :layer.out_irreps.dim].double()
            row = _grade(ref[:N - 1], got[:N - 1])
            row["all_masked_node_is_residual"] = bool(
                torch.allclose(got[0], ref[0], rtol=1e-2, atol=1e-2))
            rows[name] = row
            if not (row["finite"] and row["all_masked_node_is_residual"]
                    and row["pcc"] >= BAR["pcc_min"]
                    and row["max_rel_err"] <= BAR["max_rel_err"]
                    and row["rel_frobenius"] <= BAR["rel_frobenius"]):
                failed.append(name)
    finally:
        stop.set()
        th.join(timeout=30)
    clock = clock_summary(samples)
    record_property("diffdock_wh_grade", {"rows": rows, **clock,
                                          "chip": os.environ.get("TT_VISIBLE_DEVICES")})
    print({"rows": rows, **clock})
    assert not failed, f"over the bar {BAR} on Wormhole: " + ", ".join(
        f"{n} {rows[n]}" for n in failed) + f" (AICLK {clock})"
