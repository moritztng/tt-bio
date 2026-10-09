"""tt_bio.tfg.guidance.Guidance: rigid-mode resolution and the late pass, as upstream's sample_diffusion.

Per-step parity of Guidance.step with upstream on real 1a14 features is perf/tfg_port/guidance_1a14.py
(slow: the dense search runs on the host); here the composition is checked on the small rigid cases.
"""
import pytest
import torch

from tests.test_tfg_rigid import contact_feats, make_case
from tt_bio.tfg import epitope, rigid
from tt_bio.tfg.guidance import Guidance
from tt_bio.tfg.rigid import RigidSchedule


def _restraint_term(g):
    return next(t for t in g.cfg.terms if t.name == "UserDistanceRestraintPotential")


def test_no_constraint_is_physics_only():
    coords, feats = make_case(2)
    plain = {k: v for k, v in feats.items() if not k.startswith("user_")}
    g = Guidance(plain)
    assert g.mode == "off"
    assert _restraint_term(g).interval != 0


def test_contact_turns_rigid_on_and_atom_restraint_off():
    coords, feats = make_case(3)
    g = Guidance(contact_feats(feats))
    assert g.mode == "on"
    term = _restraint_term(g)
    assert term.interval == 0 and term.enable_projection is False


def test_epitope_needs_rigid_mode():
    coords, feats = make_case(3)
    assert epitope.active(feats)
    with pytest.raises(ValueError, match="only by rigid-body guidance"):
        Guidance(feats, schedule=RigidSchedule(mode="off"))


def test_late_pass_follows_upstream_schedule(monkeypatch):
    coords, feats = make_case(2)
    g = Guidance(contact_feats(feats))
    calls = []
    monkeypatch.setattr(g.engine, "update", lambda x_noisy, x0, **kw: x_noisy)
    monkeypatch.setattr(rigid, "search_rigid_contact", lambda x, f, core=None: calls.append("search") or x)
    monkeypatch.setattr(rigid, "refine_rigid_contact",
                        lambda x, f, iterations, core=None: calls.append(("refine", iterations)) or x)
    seen = {}
    for k in range(200):
        calls.clear()
        g.step(coords, coords, t_hat=1.0, sigma_t=0.5, eta=1.5, step=k, n_step=200)
        if calls:
            seen[k] = list(calls)
    coarse, refine = rigid.intervention_schedule(200, RigidSchedule())
    assert set(seen) == set(coarse) | set(refine)
    assert seen[190] == ["search", ("refine", 40)]
    assert seen[199] == ["search", ("refine", 120)]
    assert seen[192] == [("refine", 40)]
    assert min(seen) == 190
