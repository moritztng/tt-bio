"""`train.bridge`: a host torch parameter is an AdamW leaf that steps exactly as torch's AdamW does.

Host-only. The device half (`Bridge.up/down/backward`) is graded on a card by
`perf/r4t_tape/gradcheck_bridge.py`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("ttnn")

from tt_bio.train import optim  # noqa: E402
from tt_bio.train.bridge import HostParam, host_params  # noqa: E402


def _run(steps=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(16, 8)
    ref = torch.nn.Linear(16, 8)
    ref.load_state_dict(lin.state_dict())
    opt = optim.AdamW(host_params(lin), lr=1e-2, weight_decay=0.1, clip_norm=0.0)
    topt = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.1, eps=1e-8)
    for _ in range(steps):
        x = torch.randn(4, 16, generator=g)
        for m in (lin, ref):
            (m(x) ** 2).sum().backward()
        opt.step()
        opt.zero_grad()
        topt.step()
        topt.zero_grad()
    return lin, ref


def test_a_host_param_steps_like_torch_adamw():
    lin, ref = _run()
    for (n, a), (_, b) in zip(lin.named_parameters(), ref.named_parameters()):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), f"{n}: {(a - b).abs().max():.2e}"
        assert a.grad is None


def test_the_check_can_fail():
    """A write that never lands (the module's tensor left alone) is caught by the same comparison."""
    keep = HostParam.write
    HostParam.write = lambda self, theta: None
    try:
        lin, ref = _run()
    finally:
        HostParam.write = keep
    assert not all(torch.allclose(a, b, rtol=1e-5, atol=1e-6)
                   for a, b in zip(lin.parameters(), ref.parameters()))


def test_the_bridge_reference_agrees_with_finite_differences():
    import importlib.util
    from pathlib import Path
    from tt_bio.train import checks as CK
    spec = importlib.util.spec_from_file_location(
        "_gbr", Path(__file__).resolve().parents[1] / "perf/r4t_tape/gradcheck_bridge.py")
    GB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GB)
    rng = CK.case_rng(7, "bridge")
    r64 = {k: torch.from_numpy(v) for k, v in GB.build("bridge", rng).items()}
    wt = (torch.from_numpy(rng.standard_normal((1, GB.N, GB.DO))),) * 2
    ref = {k: v.clone().requires_grad_(k != "x") for k, v in r64.items()}
    worst, probed, _ = CK.fd_check(lambda: GB.ref_forward(ref, wt, r64), [v for k, v in ref.items() if k != "x"],
                                   n_probe=24, seed=7)
    assert probed > 0 and worst < 2e-6, (worst, probed)
