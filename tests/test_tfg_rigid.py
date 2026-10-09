"""Parity of tt_bio.tfg.rigid / tt_bio.tfg.epitope with OpenDDE v1.2.0's dense TFG path.

Live tests import upstream from TFG_UPSTREAM (default /tmp/tfgsrc/up) with OPENDDE_RIGID_CORE=off and
are skipped when the clone is absent. The fixture tests compare against upstream outputs saved in
tests/data/tfg/rigid_parity.pt; regenerate with ``python tests/test_tfg_rigid.py`` (needs the clone).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

from tt_bio.tfg import epitope as ep
from tt_bio.tfg import rigid as rc
from tt_bio.tfg.rigid import RigidSchedule

UPSTREAM = Path(os.environ.get("TFG_UPSTREAM", "/tmp/tfgsrc/up"))
FIXTURE = Path(__file__).parent / "data" / "tfg" / "rigid_parity.pt"
HAVE_UPSTREAM = (UPSTREAM / "opendde" / "tfg" / "epitope_guidance.py").is_file()
needs_upstream = pytest.mark.skipif(not HAVE_UPSTREAM, reason="OpenDDE clone not found (set TFG_UPSTREAM)")

N_ELEMENT = 128
# Backbone-like atoms per residue: N, CA, C, O (element index = atomic number - 1).
RES_ELEMENTS = (6, 5, 5, 7)


def _upstream():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import opendde.tfg.epitope_guidance as up_ep
    import opendde.tfg.rigid_contact as up_rc

    return up_rc, up_ep


@pytest.fixture
def upstream(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENDDE_RIGID"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "off")
    return _upstream()


def _globule(gen, n_res, centre, radius):
    """Residue centres inside a sphere with 4 atoms each, jittered 1.3 A around the centre."""
    pts = []
    while len(pts) < n_res:
        p = (torch.rand(3, generator=gen) * 2 - 1) * radius
        if p.norm() <= radius and all((p - q).norm() > 3.6 for q in pts):
            pts.append(p)
    res = torch.stack(pts) + torch.tensor(centre)
    off = torch.randn(n_res, 4, 3, generator=gen)
    off = 1.3 * off / off.norm(dim=-1, keepdim=True)
    return (res[:, None] + off).reshape(-1, 3)


def _rotate(x, gen):
    q = torch.linalg.qr(torch.randn(3, 3, generator=gen))[0]
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    c = x.mean(0)
    return (x - c) @ q.T + c


def make_case(n_chains, seed=0):
    """Antigen chain A at the origin plus one (2-chain) or two (heavy + light) antibody chains.

    Samples: 0 docked onto the epitope, 1 a few A short of it, 2 far off with a random orientation.
    """
    gen = torch.Generator().manual_seed(seed)
    n_ag = 50
    ab_sizes = [36] if n_chains == 2 else [22, 18]
    antigen = _globule(gen, n_ag, (0.0, 0.0, 0.0), 13.0)
    ab = [_globule(gen, n, (0.0, 0.0, 0.0), 9.0 if len(ab_sizes) == 1 else 7.5) for n in ab_sizes]
    if len(ab) == 2:
        ab[1] = ab[1] + torch.tensor([0.0, 13.5, 0.0])
    ab_all = torch.cat(ab)
    ab_all = ab_all - ab_all.mean(0)
    n_res = [n_ag] + ab_sizes
    n_atom = 4 * sum(n_res)
    chain_of_token = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_res)])
    atom_to_token = torch.arange(sum(n_res)).repeat_interleave(4)
    element = torch.tensor(RES_ELEMENTS).repeat(sum(n_res))
    ref_element = torch.nn.functional.one_hot(element, N_ELEMENT).float()
    movable = torch.zeros(n_atom, dtype=torch.bool)
    movable[4 * n_ag :] = True
    # Epitope: the 8 antigen residues with the largest x of their CA, on the +x face.
    ca = antigen.reshape(n_ag, 4, 3)[:, 1]
    epi_res = torch.topk(ca[:, 0], 8).indices.sort().values
    epi_index = (4 * epi_res[:, None] + torch.arange(4)[None]).long()
    epi_index[::3, 3] = -1  # ragged residues
    # Paratope: antibody residues on its -x face.
    ab_ca = ab_all.reshape(-1, 4, 3)[:, 1]
    para_res = torch.topk(-ab_ca[:, 0], 10).indices
    paratope = torch.zeros(n_atom, dtype=torch.bool)
    for r in para_res.tolist():
        paratope[4 * n_ag + 4 * r : 4 * n_ag + 4 * r + 4] = True
    epi_centre = antigen[epi_index[epi_index >= 0]].mean(0)
    poses = []
    for dx in (0.0, 4.0):
        shift = epi_centre + torch.tensor([15.0 + dx, 0.0, 0.0])
        poses.append(ab_all + shift)
    poses.append(_rotate(ab_all, gen) + torch.tensor([10.0, 32.0, -18.0]))
    coords = torch.stack([torch.cat([antigen, p]) for p in poses]).float()
    # Contacts: epitope CA to paratope CA pairs, window [0, 8].
    pairs = torch.stack(
        [4 * epi_res[:4] + 1, 4 * n_ag + 4 * para_res[:4] + 1]
    ).long()
    feats = {
        "asym_id": chain_of_token,
        "atom_to_token_idx": atom_to_token,
        "ref_element": ref_element,
        "user_rigid_movable_atom": movable,
        "user_distance_restraint_index": pairs,
        "user_distance_restraint_lower_bound": torch.zeros(pairs.shape[1]),
        "user_distance_restraint_upper_bound": torch.full((pairs.shape[1],), 8.0),
        "user_epitope_atom_index": epi_index,
        "user_epitope_paratope_atom": paratope,
        "user_epitope_k": torch.tensor([5]),
    }
    return coords, feats


def contact_feats(feats, with_mask=True):
    out = {k: v for k, v in feats.items() if not k.startswith("user_epitope")}
    if not with_mask:
        out.pop("user_rigid_movable_atom")
    return out


def whole_ab_feats(feats):
    out = dict(feats)
    out["user_epitope_paratope_atom"] = feats["user_rigid_movable_atom"].clone()
    return out


def dock_sample0(coords, feats):
    """Replace sample 0 by its epitope-guided pose (our pipeline) and fit K and the contact window to it.

    K becomes the number of residues that pose reaches (at most 5) and the upper bound the next
    0.5 A above its longest contact (at least 8 A), so sample 0 meets both requests and samples 1 and
    2 meet neither.
    """
    y = ep.guide_x0(coords, feats, 100)
    fixed_ids, moving_ids, epi_local, valid, para_local, _ = ep.groups(coords, feats)
    d = ep.residue_distances(y[:, moving_ids], y[:, fixed_ids], epi_local, valid, para_local)[0]
    feats = dict(feats)
    feats["user_epitope_k"] = torch.tensor([max(1, min(5, int(ep.reached_count(d)[0])))])
    _, _, fix_atoms, mov_atoms = rc.contact_groups(coords, contact_feats(feats))
    longest = (y[0, mov_atoms] - y[0, fix_atoms]).norm(dim=-1).max().item()
    upper = max(8.0, math.ceil(longest * 2 + 0.01) / 2)
    feats["user_distance_restraint_upper_bound"] = torch.full_like(
        feats["user_distance_restraint_upper_bound"], upper
    )
    out = coords.clone()
    out[0] = y[0]
    return out, feats


def build_cases():
    """{2: (coords, feats), 3: (coords, feats)} with sample 0 docked."""
    return {n: dock_sample0(*make_case(n, seed=n)) for n in (2, 3)}


def _compare(name, ours, ref, report):
    diff = (ours.float() - ref.float()).abs().max().item()
    report[name] = diff
    assert torch.equal(ours, ref), "%s: max abs diff %.3e A" % (name, diff)


CONTACT_OUTPUTS = ("refine_contact", "search_contact", "guide_x0_contact")
EPITOPE_OUTPUTS = ("refine_epitope", "search_epitope", "search_epitope_whole", "guide_x0_epitope")


def _run_all(mod_rc, mod_ep, coords, feats, names=CONTACT_OUTPUTS + EPITOPE_OUTPUTS):
    cf = contact_feats(feats)
    calls = {
        "refine_contact": lambda: mod_rc.refine_rigid_contact(coords, cf),
        "search_contact": lambda: mod_rc.search_rigid_contact(coords, cf),
        "guide_x0_contact": lambda: mod_ep.guide_x0(coords, cf, 100),
        "refine_epitope": lambda: mod_ep.refine_epitope(coords, feats),
        "search_epitope": lambda: mod_ep.search_epitope(coords, feats),
        "search_epitope_whole": lambda: mod_ep.search_epitope(coords, whole_ab_feats(feats)),
        "guide_x0_epitope": lambda: mod_ep.guide_x0(coords, feats, 103),
    }
    return {name: calls[name]() for name in names}


@pytest.fixture(scope="module")
def cases():
    return build_cases()


@needs_upstream
@pytest.mark.parametrize("names", [CONTACT_OUTPUTS, EPITOPE_OUTPUTS], ids=["contact", "epitope"])
@pytest.mark.parametrize("n_chains", [2, 3])
def test_live_parity(upstream, cases, n_chains, names):
    up_rc, up_ep = upstream
    coords, feats = cases[n_chains]
    ours = _run_all(rc, ep, coords, feats, names)
    ref = _run_all(up_rc, up_ep, coords, feats, names)
    report = {}
    for name in ours:
        _compare(name, ours[name], ref[name], report)
    _check_behaviour(coords, ours)


def _check_behaviour(coords, out):
    """Sample 0 meets both requests and must not move; the far sample 2 must be moved by the searches."""
    for name, y in out.items():
        assert torch.equal(y[0], coords[0]), name
        if name.startswith("search") or name.startswith("guide"):
            assert not torch.equal(y[2], coords[2]), name


@needs_upstream
def test_live_two_chain_without_mask(upstream, cases):
    up_rc, _ = upstream
    coords, feats = cases[2]
    cf = contact_feats(feats, with_mask=False)
    for fn in ("refine_rigid_contact", "search_rigid_contact"):
        assert torch.equal(getattr(rc, fn)(coords, cf), getattr(up_rc, fn)(coords, cf)), fn


@needs_upstream
def test_live_energies_and_helpers(upstream, cases):
    up_rc, up_ep = upstream
    coords, feats = cases[3]
    g_ours = ep.groups(coords, feats)
    g_up = up_ep.groups(coords, feats)
    for a, b in zip(g_ours, g_up):
        assert a == b if isinstance(a, int) else torch.equal(a, b)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = g_ours
    fixed, moving = coords[:, fixed_ids], coords[:, moving_ids]
    rsum = ep._radii(coords, feats, moving_ids, fixed_ids)
    assert torch.equal(rsum, up_ep._radii(coords, feats, moving_ids, fixed_ids))
    for a, b in zip(
        ep.contact_energy_and_gradient(moving, fixed, epi_local, valid, para_local, k),
        up_ep.contact_energy_and_gradient(moving, fixed, epi_local, valid, para_local, k),
    ):
        assert torch.equal(a, b)
    for a, b in zip(
        ep.candidate_energy(moving, fixed, rsum, epi_local, valid, para_local, k),
        up_ep.candidate_energy(moving, fixed, rsum, epi_local, valid, para_local, k),
    ):
        assert torch.equal(a, b)
    for grad in (False, True):
        ours = rc.clash_terms(moving, fixed, rsum, rc.SOFT_OVERLAP, want_gradient=grad)
        ref = up_rc.clash_terms(moving, fixed, rsum, rc.SOFT_OVERLAP, want_gradient=grad)
        for a, b in zip(ours, ref):
            assert (a is None and b is None) or torch.equal(a, b)
    centroid = moving.mean(1)
    chain = feats["asym_id"][feats["atom_to_token_idx"]][fixed_ids]
    a_ours, n_ours = ep.approach_axes(fixed, chain, epi_local, valid, centroid)
    a_up, n_up = up_ep.approach_axes(fixed, chain, epi_local, valid, centroid)
    assert n_ours == n_up and all(torch.equal(a, b) for a, b in zip(a_ours, a_up))
    u = torch.nn.functional.normalize(torch.randn(5, 3, generator=torch.Generator().manual_seed(1)), dim=-1)
    v = torch.cat([u[:3], -u[3:4], u[4:5]])
    assert torch.equal(ep.rotation_aligning(u, v), up_ep._rotation_aligning(u, v))
    assert torch.equal(ep.rotation_about(u, 0.7), up_ep._rotation_about(u, 0.7))
    assert torch.equal(ep.interface_patch(moving, fixed), up_ep._interface_patch(moving, fixed))
    mask_ours = rc.contact_groups(coords, contact_feats(feats))
    mask_up = up_rc.contact_groups(coords, contact_feats(feats))
    assert all(torch.equal(a, b) for a, b in zip(mask_ours, mask_up))


@needs_upstream
def test_live_chunked_clash_branch(upstream, monkeypatch, cases):
    """A small pair budget forces the chunked clash summary and severe_transition gate."""
    up_rc, up_ep = upstream
    monkeypatch.setattr(rc, "EXACT_CLASH_BUDGET", 40_000)
    monkeypatch.setattr(up_rc, "EXACT_CLASH_BUDGET", 40_000)
    coords, feats = cases[2]
    cf = contact_feats(feats)
    assert torch.equal(rc.refine_rigid_contact(coords, cf), up_rc.refine_rigid_contact(coords, cf))
    assert torch.equal(ep.refine_epitope(coords, feats), up_ep.refine_epitope(coords, feats))


SCHEDULE_ENV = [
    ({}, RigidSchedule()),
    ({"OPENDDE_RIGID_START": "150", "OPENDDE_RIGID_EVERY": "5"}, RigidSchedule(start=150, every=5)),
    ({"OPENDDE_RIGID_X0_START": "off"}, RigidSchedule(x0_start=None)),
    ({"OPENDDE_RIGID_X0_START": "20", "OPENDDE_RIGID_X0_EVERY": "4", "OPENDDE_RIGID_X0_LAST": "60"},
     RigidSchedule(x0_start=20, x0_every=4, x0_last=60)),
]


@needs_upstream
@pytest.mark.parametrize("env,schedule", SCHEDULE_ENV)
def test_live_schedules(upstream, monkeypatch, env, schedule):
    up_rc, up_ep = upstream
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for steps in (200, 191, 190, 160, 50, 1):
        try:
            ref = up_rc.intervention_schedule(steps)
        except ValueError:
            with pytest.raises(ValueError):
                rc.intervention_schedule(steps, schedule)
            continue
        assert rc.intervention_schedule(steps, schedule) == ref, steps
    assert up_ep.x0_schedule() == rc.x0_schedule(schedule)
    for step in range(0, 201):
        assert up_ep.x0_step_active(step) == rc.x0_step_active(step, schedule), step


@needs_upstream
@pytest.mark.parametrize("mode", ["auto", "on", "off", "control"])
def test_live_rigid_mode(upstream, monkeypatch, cases, mode):
    _, up_ep = upstream
    monkeypatch.setenv("OPENDDE_RIGID_CONTACT", mode)
    coords, feats = cases[3]
    variants = [
        feats,
        contact_feats(feats),
        contact_feats(feats, with_mask=False),
        contact_feats(cases[2][1], with_mask=False),
        {"asym_id": feats["asym_id"], "atom_to_token_idx": feats["atom_to_token_idx"]},
    ]
    for f in variants:
        for tfg in (True, False):
            assert ep.rigid_mode(f, tfg, RigidSchedule(mode=mode)) == up_ep.rigid_mode(f, tfg)


def test_schedule_validation():
    with pytest.raises(ValueError):
        RigidSchedule(mode="maybe")
    with pytest.raises(ValueError):
        RigidSchedule(x0_start=50, x0_last=40)
    with pytest.raises(ValueError):
        RigidSchedule(x0_every=0)
    with pytest.raises(ValueError):
        rc.intervention_schedule(100, RigidSchedule(start=190))
    assert rc.intervention_schedule(100, RigidSchedule()) == ([99], [99])
    assert rc.intervention_schedule(200, RigidSchedule())[0] == [190, 199]
    assert not any(rc.x0_step_active(s, RigidSchedule()) for s in range(100))
    assert rc.x0_step_active(187, RigidSchedule()) and not rc.x0_step_active(190, RigidSchedule())


def _fixture_cases(data):
    out = {}
    for n_chains, entry in data["cases"].items():
        feats = dict(entry["feats"])
        feats["ref_element"] = torch.nn.functional.one_hot(feats.pop("element"), N_ELEMENT).float()
        out[n_chains] = (entry["coords"], feats, entry["expected"])
    return out


@pytest.mark.skipif(not FIXTURE.is_file(), reason="fixture missing")
@pytest.mark.parametrize("names", [CONTACT_OUTPUTS, EPITOPE_OUTPUTS], ids=["contact", "epitope"])
@pytest.mark.parametrize("n_chains", [2, 3])
def test_fixture_parity(n_chains, names):
    coords, feats, expected = _fixture_cases(torch.load(FIXTURE, weights_only=True))[n_chains]
    ours = _run_all(rc, ep, coords, feats, names)
    report = {}
    for name in names:
        _compare(name, ours[name], expected[name], report)
    _check_behaviour(coords, ours)


def write_fixture():
    up_rc, up_ep = _upstream()
    os.environ["OPENDDE_RIGID_CORE"] = "off"
    data = {"cases": {}}
    for n_chains, (coords, feats) in build_cases().items():
        expected = _run_all(up_rc, up_ep, coords, feats)
        stored = {k: v for k, v in feats.items() if k != "ref_element"}
        stored["element"] = feats["ref_element"].argmax(-1)
        data["cases"][n_chains] = {"coords": coords, "feats": stored, "expected": expected}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, FIXTURE)
    print(FIXTURE, FIXTURE.stat().st_size, "bytes")


if __name__ == "__main__":
    write_fixture()
