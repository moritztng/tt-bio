"""Every float64 reference the gradient evidence rests on, checked against finite differences.

The tape's evidence is a pyramid. At the bottom sits a float64 torch reference; above it, the
device gradient scored against that reference; above that, the module and end-to-end runs. Only
the bottom layer is card-free, and until this file existed it could not run without one --
`perf/hallgrad/gradcheck.py` and `perf/train_a1_defork/gradcheck_dispatch.py` both validate their
reference with `fd_check` and then import ttnn in the same `main()`, so a box with no card ran
neither. A reference nobody re-checked is how a campaign chases a confident wrong number, and it
is the one part of the chain that costs nothing to re-check on any host.

So this gate runs `fd_check` over the union of both harnesses' cases, in float64 throughout, with
the same 2e-6 bar the harnesses use. It asserts the references, not the device. A pass here says
the thing the device is scored against is right; it says nothing about the device, which needs
the harnesses and a card.

The negative control is not a note: `test_the_gate_can_fail` perturbs a reference forward and
asserts finite differences catch it, because a check that has never failed is not known to work.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

FD_BAR = 2e-6          # both harnesses' own reference bar
N_PROBE = 24           # gradcheck_dispatch's default; enough to catch a wrong axis or term


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GC = _load("_gc_ref", "perf/hallgrad/gradcheck.py")
GD = _load("_gd_ref", "perf/train_a1_defork/gradcheck_dispatch.py")

# The op-level harness: 16 cases, every one a forward `tt_bio.autograd` implements directly.
OP_CASES = sorted(GC.CASES)
# The attach-point harness: the same ops written the way `protenix.py` writes them, plus the
# four pure-eltwise cases that exist to give the eltwise bar something to score.
SITE_CASES = ["linear", "linear_nobias", "linear_silu", "linear_relu", "linear_sigmoid",
              "layernorm", "mul", "add", "sigmoid", "silu"]


def _fd(build, forward, name, seed=7, frozen=()):
    """Central differences against torch's analytic float64 gradient. No device, ever."""
    rng = np.random.default_rng([seed, abs(hash(name)) % (2 ** 31)])
    raw = build(name, rng)
    ref = {k: torch.from_numpy(v).to(torch.float64).clone().requires_grad_(k not in frozen)
           for k, v in raw.items()}
    wt = torch.from_numpy(rng.standard_normal(tuple(forward(name, ref).shape)))

    def loss_fn():
        return (forward(name, ref) * wt).sum()

    return GC.fd_check(loss_fn, [v for k, v in ref.items() if k not in frozen],
                       n_probe=N_PROBE, seed=seed)


@pytest.mark.parametrize("name", OP_CASES)
def test_the_op_level_reference_agrees_with_finite_differences(name):
    worst, probed, eligible = _fd(lambda n, r: GC.CASES[n](r), GC.torch_forward, name,
                                  frozen=GC.FROZEN.get(name, ()))
    assert probed > 0, f"{name}: nothing was probed, so nothing was checked"
    assert worst < FD_BAR, (f"{name}: float64 reference disagrees with central differences by "
                            f"{worst:.2e} over {probed} of {eligible} resolvable coordinates")


@pytest.mark.parametrize("name", SITE_CASES)
def test_the_call_site_reference_agrees_with_finite_differences(name):
    worst, probed, eligible = _fd(GD.build, GD.ref_forward, name)
    assert probed > 0, f"{name}: nothing was probed, so nothing was checked"
    assert worst < FD_BAR, (f"{name}: float64 reference disagrees with central differences by "
                            f"{worst:.2e} over {probed} of {eligible} resolvable coordinates")


def test_frozen_lora_base_weights_are_declared_and_really_are_frozen():
    """LoRA's whole claim is that W does not move. A gradient on W is full fine-tuning."""
    assert GC.FROZEN["lora"] == ("w",)
    assert GC.FROZEN["lora_bias"] == ("w", "bias")
    rng = np.random.default_rng(7)
    t = {k: torch.from_numpy(v).to(torch.float64).requires_grad_(True)
         for k, v in GC.case_lora(rng).items()}
    GC.torch_forward("lora", t).sum().backward()
    assert t["w"].grad is not None, "sanity: torch would give W a gradient if asked"
    assert t["a"].grad.abs().max() > 0, "the adapter must move, or the case proves nothing"


class _WrongAxisNorm(torch.autograd.Function):
    """Correct value, gradient reduced over the wrong axis. Only the backward is wrong."""

    @staticmethod
    def forward(ctx, x, gamma, beta):
        ctx.shapes = (gamma.shape, beta.shape)
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        rstd = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + GD.EPS_PROD)
        ctx.save_for_backward(xc, rstd, gamma)
        return xc * rstd * gamma + beta

    @staticmethod
    def backward(ctx, g):
        xc, rstd, gamma = ctx.saved_tensors
        dn = g * gamma
        # The defect: the two correction terms reduce over dim 0 instead of the last dim.
        dx = rstd * (dn - dn.mean(0, keepdim=True)
                     - xc * rstd * (dn * xc * rstd).mean(0, keepdim=True))
        gs, bs = ctx.shapes
        return dx, (g * xc * rstd).sum(0).reshape(gs), g.sum(0).reshape(bs)


@pytest.mark.parametrize("name,forward", [
    # Both defects live in the BACKWARD, because that is what fd_check reads. A defect in the
    # forward would move the analytic gradient and the finite differences together and this
    # gate would stay green -- which is a true statement about its reach, not a weakness to
    # hide: it validates a reference against itself, and the forward is validated by the
    # parity harnesses against upstream instead.
    ("layernorm", lambda n, t: _WrongAxisNorm.apply(t["x"], t["gamma"], t["beta"])),
    # silu with its second term dropped: the value is silu, the gradient is sigmoid(y) alone.
    ("linear_silu", lambda n, t: (lambda y: y * torch.sigmoid(y).detach())(
        t["x"] @ t["w"] + t["b"])),
])
def test_the_gate_can_fail(name, forward):
    """A reference check that has never failed is not known to be able to."""
    worst, probed, _ = _fd(GD.build, forward, name)
    assert probed > 0
    assert worst >= FD_BAR, (f"{name}: the injected backward defect passed at {worst:.2e}, so "
                             f"this gate cannot detect that class of error")
