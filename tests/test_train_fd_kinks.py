"""`checks.fd_check` on a reference with ReLU kinks: shrinks the step past a kink, skips only coordinates
whose finite difference straddles one at every step, still catches a wrong analytic gradient, and refuses when nothing smooth is left."""
import torch

from tt_bio.train import checks as CK


def _relu_net(n=256, seed=0, wrong=False):
    """Many ReLU units sitting near zero, so a 1e-2 step crosses one on some probes (2 of 8 at seed 0)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(8, generator=g, dtype=torch.float64).requires_grad_(True)
    x = torch.randn(n, 8, generator=g, dtype=torch.float64)
    c = torch.randn(n, generator=g, dtype=torch.float64)

    class Relu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            ctx.save_for_backward(a)
            return a.clamp_min(0)

        @staticmethod
        def backward(ctx, gy):
            (a,) = ctx.saved_tensors
            return gy * (a > 0) * (1.3 if wrong else 1.0)

    return (lambda: (Relu.apply(x @ w) * c).sum()), [w]


def test_a_kinked_reference_is_verified_by_shrinking_the_step():
    loss, ps = _relu_net()
    kinks = []
    worst, probed, _ = CK.fd_check(loss, ps, n_probe=8, h=1e-2, kinks=kinks)
    assert probed == 8 and not kinks and worst < CK.FD_BAR


def test_without_the_kink_test_the_same_probe_fails():
    # the A arm: one central difference at h is what fd_check did before; it misreads the kink
    loss, (w,) = _relu_net()
    loss().backward()
    flat, h = w.detach().reshape(-1), 1e-2
    scale = w.grad.abs().max().item()
    err = max(abs(CK._central(loss, flat, i, h) - w.grad[i].item()) / scale for i in range(8))
    assert err > CK.FD_BAR


def test_a_wrong_gradient_still_fails_on_the_smooth_coords():
    loss, ps = _relu_net(wrong=True)
    worst, probed, _ = CK.fd_check(loss, ps, n_probe=8, h=1e-5)
    assert probed > 0 and worst > 0.1


def test_nothing_smooth_left_is_not_verified():
    loss, ps = _relu_net(n=200000)
    worst, _, _ = CK.fd_check(loss, ps, n_probe=8, h=5.0)
    assert worst == float("inf")


def _shared_kink(h, scale):
    """f(t) = g t + s(t), s odd and zero inside |t| < 5h/100: the steps h and h/10 both straddle the
    jump and share an error of 3 bars, differing by 0.2 bar (the pugc3 PairUpdate shape); h/100 is smooth."""
    g, bar = 1.0, CK.FD_BAR * scale
    E, D = 3 * bar, 0.2 * bar / (0.99 * h * h)
    t = torch.zeros(1, dtype=torch.float64)

    def f():
        x = t[0].item()
        s = 0.0 if abs(x) < 0.05 * h else (E * x + D * x ** 3)
        return torch.tensor(g * x + s, dtype=torch.float64)
    return f, t, g, bar


def test_a_shared_kink_error_is_skipped_not_accepted():
    h, scale = 1e-2, 1.0
    f, t, g, bar = _shared_kink(h, scale)
    assert CK._smooth_central(f, t, 0, h, CK.FD_AGREE * CK.FD_BAR * scale) is None


def test_the_old_agreement_tolerance_accepts_the_kinked_value():
    # the A arm: at 0.5 x FD_BAR the same pair is accepted, and the accepted value is 3 bars off
    h, scale = 1e-2, 1.0
    f, t, g, bar = _shared_kink(h, scale)
    v = CK._smooth_central(f, t, 0, h, 0.5 * CK.FD_BAR * scale)
    assert v is not None and abs(v - g) > CK.FD_BAR * scale


def _mostly_kinked(n=16, kinked=12, h=1e-2):
    """sum(a w) + sum_i<kinked relu(w_i - 0.05 h): those coordinates straddle a kink at h and h/10."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(n, generator=g, dtype=torch.float64)
    w = torch.zeros(n, dtype=torch.float64).requires_grad_(True)
    return (lambda: (a * w).sum() + torch.relu(w[:kinked] - 0.05 * h).sum()), [w]


def test_a_mostly_kinked_tensor_still_gets_its_smooth_probes():
    loss, ps = _mostly_kinked()
    kinks = []
    worst, probed, _ = CK.fd_check(loss, ps, n_probe=4, h=1e-2, kinks=kinks)
    assert probed == 4 and kinks and all(i < 12 for i in kinks) and worst < CK.FD_BAR


def test_one_draw_per_probe_leaves_the_same_tensor_unverified():
    # the A arm: drawing exactly n_probe coordinates (the old sampler) mostly lands on kinks
    loss, (w,) = _mostly_kinked()
    loss().backward()
    flat, scale = w.detach().reshape(-1), w.grad.abs().max().item()
    hits = sum(CK._smooth_central(loss, flat, i, 1e-2, CK.FD_AGREE * CK.FD_BAR * scale) is None for i in range(16))
    assert hits == 12
