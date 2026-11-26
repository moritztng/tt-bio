"""`checks.composed_verdict` (D28): a composed block passes on per-tensor max/median rel_L2 and worst cosine,
and the subtle broken arm (one tensor's gradient scaled by 1.10) fails it."""
import numpy as np

from tt_bio.train import checks as CK


def _report(scale=None, noise=0.01, n=12):
    rng = np.random.default_rng(0)
    rep = CK.GradcheckReport(name="toy", op_class="reduction", dtype="bfloat16", reference_verified=True)
    for i in range(n):
        ref = rng.standard_normal(4096)
        got = ref + noise * np.linalg.norm(ref) / 64 * rng.standard_normal(4096)
        if i == scale:
            got = got * 1.10
        rep.grads[f"p{i}"] = CK.metrics(got, ref)
    return rep


def test_a_block_at_the_device_floor_passes():
    assert CK.composed_verdict(_report())["pass"]


def test_one_tensor_scaled_by_1_10_fails():
    v = CK.composed_verdict(_report(scale=3))
    assert not v["pass"] and v["max"][0] == "p3"


def test_an_unverified_reference_never_passes():
    rep = _report()
    rep.reference_verified = False
    assert not CK.composed_verdict(rep)["pass"]
