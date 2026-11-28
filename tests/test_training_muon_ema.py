"""Muon routing and the EMA, against torch's own optimizers under injected gradients. No card.

RFD4-Proteina trains with ``torch.optim.Muon`` on its 2-D weights and ``torch.optim.Adam`` on
the rest, each clipped to norm 1.0 on its own, Adam warmed up linearly, and an
``AveragedModel`` EMA at 0.999. These are state-free factors of the update rule, so they are
verified exactly here (OF3T method: the optimizer is proven on CPU, only the gradient needs
a card).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch.optim, "Muon"):
    pytest.skip("torch.optim.Muon needs torch >= 2.9", allow_module_level=True)

from tt_bio.train import optim  # noqa: E402
from tt_bio.train.optim import EMA, AdamW, Muon, linear_warmup  # noqa: E402


class _Val(np.ndarray):
    def device(self):
        return None


class _P:
    def __init__(self, arr):
        self.value = np.asarray(arr, np.float32).view(_Val)
        self.grad = None


SHAPES = {"blk.w_qkv": (24, 40), "blk.w_out": (40, 16), "blk.norm.weight": (24,),
          "head.ca_linear.weight": (3, 24), "blk.bias": (40,)}
EXCLUDE = (r"^head\.",)


def _run(steps=6, lr=2e-2, warm=3, broken=False, seed=0):
    rng = np.random.default_rng(seed)
    init = {n: rng.normal(size=s).astype(np.float32) for n, s in SHAPES.items()}
    grads = [{n: (rng.normal(size=s) * (3.0 if k % 2 else 0.2)).astype(np.float32)
              for n, s in SHAPES.items()} for k in range(steps)]

    # ours
    ps = {n: _P(v) for n, v in init.items()}
    mu = Muon(exclude=EXCLUDE, weight_decay=0.0 if broken else 0.1)
    opt = AdamW(ps, lr=lr, weight_decay=0.0, clip_norm=1.0, schedule=linear_warmup(lr, warm),
                muon=mu)
    ema = EMA(0.9)
    optim_to_device = optim.to_device
    optim.to_device = lambda arr, dev, dtype=None: np.array(arr, np.float32).view(_Val)
    try:
        for g in grads:
            for n, p in ps.items():
                p.grad = g[n]
            opt.step()
            ema.update(opt.master)
    finally:
        optim.to_device = optim_to_device

    # torch, as rfproteina builds and steps it
    tp = {n: torch.nn.Parameter(torch.from_numpy(v.copy())) for n, v in init.items()}
    model = torch.nn.Module()
    for i, (n, p) in enumerate(tp.items()):
        model.register_parameter(f"p{i}", p)
    muon_p = [tp[n] for n in SHAPES if mu.takes(n, SHAPES[n])]
    adam_p = [tp[n] for n in SHAPES if not mu.takes(n, SHAPES[n])]
    t_muon = torch.optim.Muon(muon_p, lr=lr, adjust_lr_fn="match_rms_adamw")
    t_adam = torch.optim.Adam(adam_p, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(t_adam, lambda s: min((s + 1) / warm, 1.0))
    avg = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True)
    for g in grads:
        for n, p in tp.items():
            p.grad = torch.from_numpy(g[n].copy())
        for group in (muon_p, adam_p):
            torch.nn.utils.clip_grad_norm_(group, 1.0)
        t_muon.step()
        t_adam.step()
        sched.step()
        avg.update_parameters(model)
    ours = {n: opt.master[n] for n in SHAPES}
    theirs = {n: tp[n].detach().numpy() for n in SHAPES}
    ema_theirs = dict(zip(SHAPES, (p.detach().numpy() for p in avg.module.parameters())))
    return ours, theirs, ema.shadow, ema_theirs, opt


def _rel(a, b):
    return max(float(np.abs(a[n] - b[n]).max() / np.abs(b[n]).max()) for n in b)


def test_routing_matches_upstream_split():
    mu = Muon(exclude=EXCLUDE)
    assert [n for n, s in SHAPES.items() if mu.takes(n, s)] == ["blk.w_qkv", "blk.w_out"]


# The bar is the float32 floor over 6 steps, measured at 1.2e-07 (one ulp at |w| ~ 1). It is
# not a long-trajectory bar on purpose: Newton-Schulz casts to bf16, so a one-ulp difference
# in its input becomes a 2^-8 difference in one update element, and by ~20 steps two correct
# implementations sit 2e-4 apart (measured). Upstream's own GPU run differs from its CPU run
# the same way, so a long-horizon trajectory is not a parity metric for Muon.
BAR = 1e-6


def test_muon_adam_trajectory_matches_torch():
    ours, theirs, _, _, opt = _run()
    assert opt.muon_names == {"blk.w_qkv", "blk.w_out"}
    assert _rel(ours, theirs) < BAR, _rel(ours, theirs)


def test_ema_matches_averaged_model():
    _, _, ema, ema_theirs, _ = _run()
    assert _rel(ema, ema_theirs) < BAR


def test_trajectory_check_can_fail():
    """Muon without upstream's default 0.1 weight decay must not pass the same bar."""
    ours, theirs, _, _, _ = _run(broken=True)
    assert _rel(ours, theirs) > 100 * BAR


def test_no_muon_is_plain_adamw():
    rng = np.random.default_rng(1)
    init = {n: rng.normal(size=s).astype(np.float32) for n, s in SHAPES.items()}
    g = {n: rng.normal(size=s).astype(np.float32) for n, s in SHAPES.items()}
    out = []
    saved = optim.to_device
    optim.to_device = lambda arr, dev, dtype=None: np.array(arr, np.float32).view(_Val)
    try:
        for muon in (None, Muon(exclude=(r".*",))):     # excluding everything = Adam only
            ps = {n: _P(v) for n, v in init.items()}
            o = AdamW(ps, lr=1e-2, weight_decay=0.0, clip_norm=1.0, muon=muon)
            for p_n, p in ps.items():
                p.grad = g[p_n]
            o.step()
            out.append({n: o.master[n].copy() for n in SHAPES})
    finally:
        optim.to_device = saved
    for n in SHAPES:
        assert np.array_equal(out[0][n], out[1][n])
