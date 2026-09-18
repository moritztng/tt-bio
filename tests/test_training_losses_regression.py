"""Regression pins for the eight AF3 training loss terms. Host-only: numpy, no ttnn, no device.

WHAT THIS PROVES AND WHAT IT DOES NOT. `tt_bio/train/losses.py` is 387 lines of the most
load-bearing arithmetic in the training effort -- every gradient the stack produces is a
gradient of these terms -- and until this file existed the repo's test suite had no coverage
of it at all. Its CORRECTNESS is established elsewhere and not here:
`perf/ptxft/losscheck.py` runs ByteDance's own `nn.Module` for each term on CPU against these
functions on shared random inputs, in float64 both sides. That check needs the upstream
`protenix` package installed, so it cannot run in CI.

This file is the cheap half that can: golden values on fixed inputs, so a refactor that
changes a reduction, a mask convention or an epsilon fails immediately instead of silently
training a subtly wrong model. A golden pin cannot tell you the value was ever right -- only
that it has not moved. Treat a failure here as "something changed, go find out what", never as
"the loss is wrong", and treat a pass as "nothing moved", never as parity.

Two assertions in here are genuinely independent of the implementation and are the reason this
file is not purely self-referential:

  * `edm_scale` has a closed form, so it is checked against arithmetic rather than a pin.
  * `LOSS_WEIGHTS` is checked against the weights `train-r1-protenix` read out of upstream's
    `configs/configs_base.py:401-407`, which is a second source rather than this code.

NEGATIVE CONTROLS RUN AS TESTS, not as a note. `test_the_pins_are_sensitive` below perturbs a
constant in a COPY of losses.py on disk, imports it, and asserts the pins move. On disk because
the copy has to be compiled with the change: the grids and epsilons are captured as default
argument values at def-time, so `setattr(L, "DISTOGRAM_GRID", ...)` leaves the already-bound
default alone and every pin still passes. The first version of these controls was built that
way and proved nothing, which is why they are now executed rather than described. The test also
asserts the substitution applied and that the module it scored is the perturbed copy, so it
cannot pass vacuously. Pattern adopted from `train-b2-abb3-port`'s
`tests/test_abodybuilder3_reference.py`, which does the same thing better than the hand-run
version this file used to rely on.

One perturbation deliberately moves nothing: `SIGMA_DATA` 16.0 -> 16.1 breaks no pin, because
no pinned term uses it. That is why `test_edm_scale_matches_its_closed_form` asserts
`SIGMA_DATA == 16.0` literally — the closed form is a relationship and stays self-consistent at
any sigma_data, so only the value pin catches a drift in it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tt_bio.train import losses as L

SEED = 20260918
N_TOKENS = 12
N_ATOMS = 30
PKG_TRAIN = Path(L.__file__).resolve().parent

# Pinned from tt_bio/train/losses.py on wk/train-orchestrator, 2026-09-18. `.grad_abssum` is
# sum(|grad|) over the analytic gradient the term returns alongside its loss, so a backward
# regression is caught as well as a forward one.
GOLDEN = {
    "distogram": 4.612053161432,
    "distogram.grad_abssum": 1.966816881079,
    "pde": 4.679850801368,
    "pde.grad_abssum": 1.972713413767,
    "pae": 4.671819083795,
    "pae.grad_abssum": 1.969452854191,
    "mse": 0.607381310767,
    "mse.grad_abssum": 1.281032702891,
    "lddt_sum": 239.500000000000,
    "lddt_weight_sum": 330.000000000000,
    "plddt": 4.400630266549,
    "plddt.grad_abssum": 1.969263425746,
    "resolved": 0.662692589767,
    "resolved.grad_abssum": 0.842975101884,
    "bond": 1.271813035873,
    "bond.grad_abssum": 1.766697762136,
    "smooth_lddt": 0.338313077168,
    "smooth_lddt.grad_abssum": 0.167583626450,
}


def _measure() -> dict[str, float]:
    """Every term on one deterministic draw. The draw order is part of the fixture."""
    rng = np.random.default_rng(SEED)
    n, a = N_TOKENS, N_ATOMS
    out: dict[str, float] = {}

    def rec(name, v):
        if isinstance(v, tuple):
            out[name] = float(np.asarray(v[0]))
            out[f"{name}.grad_abssum"] = float(np.abs(np.asarray(v[1], np.float64)).sum())
        else:
            out[name] = float(np.asarray(v))

    true_xyz = rng.normal(0, 8, (n, 3))
    pred_xyz = true_xyz + rng.normal(0, 1.0, (n, 3))
    coord_mask = np.ones(n, bool)
    coord_mask[-2:] = False

    rec("distogram", L.distogram(rng.normal(0, 1, (n, n, 64)), true_xyz, coord_mask))
    rec("pde", L.pde(rng.normal(0, 1, (n, n, 64)), pred_xyz, true_xyz, coord_mask))
    frame_atom_index = np.stack(
        [np.arange(n), (np.arange(n) + 1) % n, (np.arange(n) + 2) % n], -1)
    rec("pae", L.pae(rng.normal(0, 1, (n, n, 64)), pred_xyz, true_xyz, coord_mask,
                     frame_atom_index))

    atom_true = rng.normal(0, 8, (a, 3))
    atom_pred = atom_true + rng.normal(0, 0.8, (a, 3))
    atom_mask = np.ones(a, bool)
    atom_mask[-3:] = False
    rec("mse", L.mse(atom_pred, atom_true, atom_mask))

    lddt, weight = L.atom_bespoke_lddt(atom_pred, atom_true, np.zeros(a, bool),
                                       np.ones(a, bool), np.ones(a, bool))
    rec("lddt_sum", np.sum(lddt))
    rec("lddt_weight_sum", np.sum(weight))
    rec("plddt", L.plddt(rng.normal(0, 1, (a, 50)), lddt, weight))
    rec("resolved", L.resolved(rng.normal(0, 1, (a, 2)), atom_mask, np.ones(a)))

    pred_dist = np.linalg.norm(atom_pred[:, None] - atom_pred[None], axis=-1)
    true_dist = np.linalg.norm(atom_true[:, None] - atom_true[None], axis=-1)
    bond_mask = (np.abs(np.arange(a)[:, None] - np.arange(a)[None]) == 1).astype(np.float64)
    rec("bond", L.bond(pred_dist, true_dist, bond_mask))

    pair_mask = (atom_mask[:, None] & atom_mask[None]).astype(np.float64)
    np.fill_diagonal(pair_mask, 0.0)
    rec("smooth_lddt", L.smooth_lddt(pred_dist, true_dist, pair_mask))
    return out


@pytest.fixture(scope="module")
def measured():
    return _measure()


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_loss_term_has_not_moved(measured, key):
    """A pin, not a parity check. See the module docstring before 'fixing' a failure."""
    assert key in measured, f"{key} disappeared from the measurement set"
    got, want = measured[key], GOLDEN[key]
    assert got == pytest.approx(want, rel=1e-9, abs=1e-11), (
        f"{key}: {got!r} vs pinned {want!r}. Something in tt_bio/train/losses.py changed the "
        f"value of this term. That may be a deliberate fix -- if so, re-derive the pin AND "
        f"re-run perf/ptxft/losscheck.py against Protenix's own module, because this file "
        f"cannot tell a correction from a regression."
    )


def test_every_term_is_covered(measured):
    """A new loss term must arrive with a pin, or this fails and asks for one."""
    missing = sorted(set(measured) - set(GOLDEN))
    assert not missing, f"measured but unpinned: {missing}. Add them to GOLDEN."
    eight = {"distogram", "smooth_lddt", "bond", "mse", "plddt", "pde", "pae", "resolved"}
    assert eight <= set(measured), f"a headline term is not exercised: {sorted(eight - set(measured))}"


def test_edm_scale_matches_its_closed_form():
    """Independent of the implementation: (sigma^2 + sd^2) / (sd * sigma)^2, loss.py:1636-1639."""
    sd = L.SIGMA_DATA
    assert sd == 16.0, f"SIGMA_DATA moved to {sd}; generator.py:40 says 16.0"
    for sigma in (0.25, 1.5, 7.0, 40.0):
        want = (sigma ** 2 + sd ** 2) / (sd * sigma) ** 2
        assert float(np.asarray(L.edm_scale(sigma))) == pytest.approx(want, rel=1e-12)
    # and the value the rest of the suite quotes, by hand: 258.25 / 576
    assert float(np.asarray(L.edm_scale(1.5))) == pytest.approx(258.25 / 576.0, rel=1e-12)


def test_loss_weights_match_upstreams_config():
    """Checked against train-r1-protenix's reading of configs/configs_base.py:401-407.

    Upstream expresses these as products: alpha_diffusion 4.0, alpha_distogram 3e-2,
    alpha_confidence 1e-4, alpha_except_pae 1.0, and per-stage alpha_pae 0 -> 1.0,
    alpha_bond 0 -> 1.0, smooth_lddt 1.0 -> 0. This asserts the products, so a drift in
    either the factors or the staging shows up here.
    """
    a_diff, a_dist, a_conf, a_except = 4.0, 3e-2, 1e-4, 1.0
    expect = {
        "pretrain": {"mse": a_diff, "smooth_lddt": a_diff * 1.0, "bond": a_diff * 0.0,
                     "distogram": a_dist, "plddt": a_conf * a_except, "pde": a_conf * a_except,
                     "resolved": a_conf * a_except, "pae": a_conf * 0.0},
        "finetune": {"mse": a_diff, "smooth_lddt": a_diff * 0.0, "bond": a_diff * 1.0,
                     "distogram": a_dist, "plddt": a_conf * a_except, "pde": a_conf * a_except,
                     "resolved": a_conf * a_except, "pae": a_conf * 1.0},
    }
    assert set(L.LOSS_WEIGHTS) == set(expect), (
        f"stages changed: {sorted(L.LOSS_WEIGHTS)} vs {sorted(expect)}")
    for stage, terms in expect.items():
        assert set(L.LOSS_WEIGHTS[stage]) == set(terms), (
            f"{stage} terms changed: {sorted(L.LOSS_WEIGHTS[stage])} vs {sorted(terms)}")
        for term, want in terms.items():
            got = L.LOSS_WEIGHTS[stage][term]
            assert got == pytest.approx(want, rel=1e-12), (
                f"LOSS_WEIGHTS[{stage!r}][{term!r}] is {got} but upstream's factors give {want}")

# --------------------------------------------------------- the controls, executed

# (anchor in losses.py, replacement, keys whose pins MUST move). The anchors are exact source
# lines, so a refactor that renames them fails this test loudly rather than silently weakening it.
PERTURBATIONS = [
    ("DISTOGRAM_GRID = (2.3125, 21.6875, 64)", "DISTOGRAM_GRID = (2.3125, 22.0, 64)",
     {"distogram", "distogram.grad_abssum"}),
    ("PLDDT_GRID = (0.0, 1.0, 50)", "PLDDT_GRID = (0.0, 1.0, 51)",
     {"plddt", "plddt.grad_abssum"}),
    ("def smooth_lddt(pred_dist, true_dist, lddt_pair_mask, *, eps=1e-10):",
     "def smooth_lddt(pred_dist, true_dist, lddt_pair_mask, *, eps=1e-6):",
     {"smooth_lddt", "smooth_lddt.grad_abssum"}),
]


def _load_perturbed(old: str, new: str):
    """Compile a copy of losses.py with `old` replaced by `new`, inside the package.

    Inside the package so its `from __future__` and any relative imports resolve the same way,
    and removed in `finally` so a failure cannot leave it behind.
    """
    src_path = PKG_TRAIN / "losses.py"
    src = src_path.read_text()
    assert old in src, (
        f"anchor not found in losses.py, so this control would test nothing: {old!r}. "
        f"If the line was legitimately renamed, update PERTURBATIONS."
    )
    perturbed = src.replace(old, new, 1)
    assert perturbed != src, "substitution did not apply"
    tmp = PKG_TRAIN / f"_losses_control_{abs(hash(old)) % 10**8}.py"
    tmp.write_text(perturbed)
    try:
        spec = importlib.util.spec_from_file_location(
            f"tt_bio.train.{tmp.stem}", tmp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # the module under test really is the perturbed copy, not the real one
        assert Path(mod.__file__).name == tmp.name, (
            f"scored {mod.__file__}, not the perturbed copy")
        yield mod
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.parametrize("old,new,must_move", PERTURBATIONS,
                         ids=[p[0].split("(")[0].split("=")[0].strip() for p in PERTURBATIONS])
def test_the_pins_are_sensitive(measured, old, new, must_move):
    """A pin that cannot fail is not a pin. Perturb the source; the named pins must move."""
    gen = _load_perturbed(old, new)
    mod = next(gen)
    try:
        global L
        real, L = L, mod
        try:
            got = _measure()
        finally:
            L = real
    finally:
        for _ in gen:
            pass

    moved = {k for k in GOLDEN
             if got[k] != pytest.approx(GOLDEN[k], rel=1e-9, abs=1e-11)}
    missing = must_move - moved
    assert not missing, (
        f"perturbing {old!r} -> {new!r} did NOT move {sorted(missing)}. Those pins are not "
        f"sensitive to it, so they are not protecting what this file claims they protect."
    )


def test_sigma_data_moves_no_pin_which_is_why_the_closed_form_asserts_it():
    """The documented exception, executed rather than asserted in prose."""
    gen = _load_perturbed("SIGMA_DATA = 16.0", "SIGMA_DATA = 16.1")
    mod = next(gen)
    try:
        global L
        real, L = L, mod
        try:
            got = _measure()
        finally:
            L = real
    finally:
        for _ in gen:
            pass
    moved = {k for k in GOLDEN
             if got[k] != pytest.approx(GOLDEN[k], rel=1e-9, abs=1e-11)}
    assert not moved, (
        f"SIGMA_DATA now reaches {sorted(moved)}. If a pinned term started using it that is "
        f"fine, but then this test is obsolete and the closed-form test carries less weight "
        f"than its docstring claims."
    )
    assert mod.SIGMA_DATA == 16.1, "control did not apply"
