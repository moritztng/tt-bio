"""RFD3_DIT_TRACE: the DiT replayed from its trace is the eager DiT bit for bit at the page's token count, over replays
whose inputs arrive at new addresses, and an eager tensor held across a replay comes back untouched."""
import os
import pathlib

import pytest
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor

pytestmark = pytest.mark.device

WEIGHTS = (pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
           / "diffusion_module.real_weights.pt")


def _inputs(seed, n=685):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, n, 768, generator=g)
    s = torch.randn(1, n, 384, generator=g)
    z = torch.randn(1, n, n, 128, generator=g) * 0.5
    idx = torch.randint(0, n, (1, n, 32), generator=g)
    return a, s, z, idx


def test_dit_trace_matches_eager(monkeypatch):
    if not WEIGHTS.exists():
        pytest.skip(f"no RFD3 weights at {WEIGHTS}")
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    dev = get_device(trace="rfd3")
    from tt_bio.rfd3 import model as M
    dit = M.build_diffusion_module(torch.load(WEIGHTS, map_location="cpu", weights_only=True)).diffusion_transformer
    sets = [_inputs(seed) for seed in (1, 2, 3, 2, 1)]

    # Traced first, on a cold program cache: the full fold met a program that only compiled at the capture.
    monkeypatch.setattr(M, "_DIT_TRACE", True)
    want = torch.arange(32 * 1024.0).reshape(32, 1024)
    got, held = [], None
    for x in sets:          # eager until warm, capture + replay, then replays at moved input addresses
        got.append(dit(*x))
        if held is None and dit._trace.get("tid") is not None:   # alive across the replays that follow
            held = ttnn.from_torch(want, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    assert held is not None and len(got) - dit._trace["eager"] >= 2, ("capture refused", dit._trace)

    monkeypatch.setattr(M, "_DIT_TRACE", False)
    ref = [dit(*x) for x in sets]
    for i, (r, o) in enumerate(zip(ref, got)):
        assert torch.equal(r, o), (i, (r - o).abs().max().item())
    assert torch.equal(ttnn.to_torch(held), want)
    dit.release_trace()
