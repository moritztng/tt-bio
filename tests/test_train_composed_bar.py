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


def _floored(site, floor_rel, got_rel, scale=None):
    """One tensor at ``site`` sitting ``got_rel`` off, beside 11 at the device floor."""
    rep = _report(scale=scale)
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(4096)
    e = rng.standard_normal(4096)
    e *= np.linalg.norm(ref) / np.linalg.norm(e)
    rep.grads[site] = CK.metrics(ref + got_rel * e, ref)
    f = CK.metrics(ref + floor_rel * e, ref)
    return rep, {site: {"rel": f["rel_l2"], "cos": f["cos"]}}


SITE = "pair_update_layers.0.transition_out.linear_1.weight"


def test_a_relu_site_inside_its_floor_passes_only_with_the_floor():
    rep, floor = _floored(SITE, 0.12, 0.15)
    assert not CK.composed_verdict(rep)["pass"]
    v = CK.composed_verdict(rep, floor)
    assert v["pass"] and SITE in v["floored"]


def test_the_floor_never_widens_a_tensor_off_the_site_list():
    name = "pair_update_layers.0.transition_out.linear_2.weight"
    rep, floor = _floored(name, 0.12, 0.15)
    v = CK.composed_verdict(rep, floor)
    assert not v["pass"] and name in v["fails"] and not v["floored"]


def test_a_relu_site_past_1_5x_its_floor_fails():
    rep, floor = _floored(SITE, 0.08, 0.15)
    v = CK.composed_verdict(rep, floor)
    assert not v["pass"] and v["fails"] == [SITE]
