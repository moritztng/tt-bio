"""The resume path, because the reproduction will be interrupted and nothing tested this.

PLAN.md §7g: the ABodyBuilder3 reproduction is a 9-to-13-day 4-chip run, and neither QuietBox
stays up that long. qb2 took 5 watchdog resets in a single day while running 4 cards at full
tilt -- exactly this run's load profile -- and each reset kills a process that has no
crash-safety of its own. So the run resumes from a checkpoint on the order of 9 to 47 times,
and the resume path is load-bearing in a way it has never been before.

It was also completely untested. Track C shipped `tt_bio/train/checkpoint.py` with
`save_adapter`, `load_adapter` and `Checkpointer`, and its 29 interface tests cover none of
them. The specific hazard is in the signature: `load_adapter(path, params, device, *, opt=None)`
takes the optimizer **optionally**. A resume written as `load_adapter(path, params, device)`
loads the weights, silently leaves both Adam moments at whatever the fresh optimizer had, and
resets the step count that drives bias correction and the LR schedule -- then trains on and
produces a plausible loss curve for a different optimisation problem. Over 193,512 steps with
tens of resumes, that is a corrupted run that looks healthy.

So this gate asserts both halves: with `opt=` everything comes back exactly, and without it the
omission really is silent. The second half is the negative control -- without it the first could
pass because the loader is lenient rather than because it is correct.

Device-free by construction. `save_adapter`/`load_adapter` keep the masters and both moments as
numpy (`safetensors.numpy`) because `ttnn.moreh_adamw` refuses an fp32 master, so the state at
risk is host-side. The only device seam is the `to_device` call that pushes a master back into a
parameter, and that is monkeypatched -- the point of the test is the optimizer state, not the
transport.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "tt_bio" / "train" / "checkpoint.py"

pytestmark = pytest.mark.skipif(
    not CKPT.is_file(),
    reason="tt_bio/train/checkpoint.py is not on this tree yet (the training interface has "
           "not landed), so there is no resume path to check")

np = pytest.importorskip("numpy")
pytest.importorskip("safetensors", reason="safetensors is in the tt-bio wheel's venv; run this "
                                          "gate with that interpreter")


SPLIT = "|"
NAMES = ["blk.0.lora_a", "blk.0.lora_b"]


def _written_state(rng):
    """The state a mid-run checkpoint would hold: masters moved, moments non-zero."""
    master = {n: rng.normal(size=(4, 3)).astype(np.float32) for n in NAMES}
    exp_avg = {n: rng.normal(size=(4, 3)).astype(np.float32) for n in NAMES}
    exp_avg_sq = {n: np.abs(rng.normal(size=(4, 3))).astype(np.float32) for n in NAMES}
    return master, exp_avg, exp_avg_sq


def _write_adapter(path, master, exp_avg, exp_avg_sq, steps):
    """Write the file in `save_adapter`'s own documented layout."""
    from safetensors.numpy import save_file
    tensors = {}
    for n in master:
        tensors[f"master{SPLIT}{n}"] = master[n]
        tensors[f"exp_avg{SPLIT}{n}"] = exp_avg[n]
        tensors[f"exp_avg_sq{SPLIT}{n}"] = exp_avg_sq[n]
    opt_state = {"steps": steps, "lr": 3e-4, "beta1": 0.9, "beta2": 0.999}
    save_file(tensors, str(path),
              metadata={"optimizer": json.dumps(opt_state), "tt_bio_adapter": "1"})


class _Param:
    """Stands in for an `ag.Tensor` parameter: `load_adapter` only assigns to `.value`."""

    class _V:
        dtype = "bf16-stand-in"

    def __init__(self):
        self.value = _Param._V()


class _Opt:
    """Stands in for `AdamW` across the surface `load_adapter` actually touches.

    `AdamW.__init__` calls `to_host` on device tensors, so it is not constructible here; what
    the resume path uses is the three dicts and `load_state_dict`, which is all of this.
    """

    def __init__(self):
        self.master, self.exp_avg, self.exp_avg_sq = {}, {}, {}
        self.steps = 0
        self.lr = 0.0

    def load_state_dict(self, d):
        self.steps = d["steps"]
        self.lr = d["lr"]


@pytest.fixture()
def loaded(monkeypatch, tmp_path):
    """`load_adapter` with the device seam replaced, returning what it restored."""
    from tt_bio.train import checkpoint as C

    seen = {}
    monkeypatch.setattr(C, "to_device", lambda arr, device, dtype=None: ("on-device", arr))

    def run(*, pass_opt):
        rng = np.random.default_rng(20260918)
        master, exp_avg, exp_avg_sq = _written_state(rng)
        path = tmp_path / ("with_opt.st" if pass_opt else "no_opt.st")
        _write_adapter(path, master, exp_avg, exp_avg_sq, steps=12345)
        params = {n: _Param() for n in NAMES}
        opt = _Opt()
        C.load_adapter(path, params, device=None, opt=(opt if pass_opt else None))
        return dict(written=(master, exp_avg, exp_avg_sq), params=params, opt=opt)

    seen["run"] = run
    return run


def test_resume_restores_the_masters_and_BOTH_adam_moments_and_the_step(loaded):
    got = loaded(pass_opt=True)
    master, exp_avg, exp_avg_sq = got["written"]
    opt = got["opt"]

    for n in NAMES:
        np.testing.assert_array_equal(
            opt.master[n], master[n],
            err_msg=f"{n}: the fp32 master did not come back exactly. The master is the "
                    f"authoritative value and the device weight is a rounded view of it, so any "
                    f"loss here decays the adapter on every resume.")
        np.testing.assert_array_equal(
            opt.exp_avg[n], exp_avg[n],
            err_msg=f"{n}: exp_avg (Adam's first moment) did not come back. A resume that "
                    f"restarts the momentum trains a different optimisation problem.")
        np.testing.assert_array_equal(
            opt.exp_avg_sq[n], exp_avg_sq[n],
            err_msg=f"{n}: exp_avg_sq (second moment) did not come back, so Adam's per-parameter "
                    f"step size restarts from scratch.")

    assert opt.steps == 12345, (
        f"the step count came back as {opt.steps}, not 12345. It drives bias correction and the "
        f"LR schedule, so a resume that resets it re-warms up in the middle of a run.")
    # And the parameters themselves were pushed back through the device seam.
    for n in NAMES:
        assert got["params"][n].value[0] == "on-device", f"{n} was never assigned"


def test_the_negative_control_omitting_opt_really_is_silent(loaded):
    """Without `opt=`, the moments and step must be demonstrably NOT restored.

    This is what makes the test above meaningful rather than merely green: it establishes that
    the loader distinguishes the two calls, so a resume that forgets the optimizer is a real
    defect and not something the loader quietly fixes.
    """
    got = loaded(pass_opt=False)
    opt = got["opt"]
    assert opt.master == {} and opt.exp_avg == {} and opt.exp_avg_sq == {}, (
        "omitting `opt=` restored optimizer state anyway. If the loader has started doing this, "
        "this gate's premise is gone and the hazard it documents no longer exists -- delete it "
        "rather than weaken it.")
    assert opt.steps == 0, "omitting `opt=` restored the step count anyway; see above."
    # The weights DID load, which is exactly why the omission is dangerous: the run looks fine.
    for n in NAMES:
        assert got["params"][n].value[0] == "on-device", (
            f"{n} did not load without `opt=`, which would make the omission loud rather than "
            f"silent. The whole hazard is that the weights arrive and the optimizer does not.")
