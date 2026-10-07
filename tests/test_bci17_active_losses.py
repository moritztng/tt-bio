"""`active_losses` must refuse the loss wirings that return a silently zero gradient.

The #17 A/B ran for hours against an empty loss set. `build_losses` returned `{}`, the design loss
was the constant 0, and the binder sequence gradient was exactly zero at every temperature, every
one_hot weight and in float32 as well as float16. Nothing raised: `weighted_design_loss`
(loss.py:115) seeds its sum with a constant `0.0` and skips a loss whose states are absent, so the
call returned a correct-looking forward -- i_pTM 0.7290, responding properly to temperature --
beside a gradient that measured nothing. Three wrong hypotheses were chased before the cause.

These tests pin the two refusals and the one reconstruction that prevent a repeat.
"""
import importlib.util
import pathlib

import pytest

MODULE = (pathlib.Path(__file__).resolve().parents[1]
          / "perf" / "bci_accept" / "harden_forward_ab.py")

#: The settings the A/B builds by hand. Deliberately the raw form, with no `weights_<loss>` keys:
#: `build_design_settings` stores it verbatim (settings.py:655), which is the trap.
SETTINGS = {
    "targets": [{"name": "hPDL1", "target_path": "/nonexistent/hPDL1.pdb",
                 "chains": "A", "hotspots": "54,56,66,115"}],
    "binder_lengths": [60, 60],
    "max_trajectories": 1, "number_of_final_designs": 1, "trajectory_only": True,
    "design_models": 1, "campaign_seed": 42,
}

#: What the harden-entry capture actually holds: one state, keyed by TARGET name, because
#: trajectory.py:145 hands `update_sequence` the per-round `active_protein_states`.
STATES = {"hPDL1"}


def load():
    spec = importlib.util.spec_from_file_location("harden_forward_ab", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_settings_would_have_no_losses():
    """The exact bug: the hand-built dict alone yields nothing for build_losses to weight."""
    from bindcraft.loss import build_losses
    from bindcraft.settings import build_design_settings
    assert build_losses(build_design_settings(SETTINGS).settings) == {}, (
        "if this ever becomes non-empty the trap is gone and active_losses can be simplified")


def test_active_losses_merges_defaults_and_returns_a_real_set():
    """`load_settings` is what puts DEFAULT_SETTINGS underneath, so losses actually exist."""
    losses = load().active_losses(SETTINGS, STATES)
    assert losses, "an empty set here is the zero-gradient bug"
    # The interface terms are the ones satisfiable by a single target-named state.
    assert "iptm_loss" in losses


def test_losses_needing_an_absent_state_are_dropped_not_raised():
    """af2.py:369 would KeyError on these inside a traced function; drop them where it is sayable."""
    losses = load().active_losses(SETTINGS, STATES)
    for name, entry in losses.items():
        assert entry.required_states <= STATES, f"{name} would KeyError in sequence_design_loss"


def test_a_state_agnostic_loss_survives_any_state_set():
    """A loss with EMPTY required_states is state-agnostic by BindCraft 2's own rule.

    `loss.py:57` leaves `required_states` empty whenever `prediction_state` is not configured
    explicitly, and `af2.py:369` then reads `entry.required_states or prediction_arrays.keys()`,
    i.e. it applies the loss to whatever states the round produced. So the drop rule must not
    treat an empty requirement as unsatisfiable -- the interface terms are exactly that, and
    dropping them would empty the objective of the only losses this comparison has.
    """
    losses = load().active_losses(SETTINGS, {"a_state_no_loss_names"})
    assert sorted(losses) == ["interface_contacts", "interface_pae", "iptm_loss"]
    assert all(not entry.required_states for entry in losses.values())


def test_every_loss_dropped_is_a_refusal():
    """When every loss in play names a state that is absent, stop rather than return nothing."""
    # These two are the default-configured ones that keep required_states=('complex',), which a
    # target-named state set does not satisfy.
    with pytest.raises(SystemExit, match="every loss was dropped"):
        load().active_losses(SETTINGS, STATES, captured={"plddt_loss", "binder_pae"})


def test_captured_active_set_wins_over_the_reconstruction():
    """The recorded active set is the captured article; the drop rule only approximates it."""
    losses = load().active_losses(SETTINGS, STATES, captured={"iptm_loss"})
    assert sorted(losses) == ["iptm_loss"]


def test_a_captured_loss_the_settings_cannot_build_is_a_refusal():
    """Silently ignoring it would compare two arms on an objective neither capture used."""
    with pytest.raises(SystemExit, match="does not"):
        load().active_losses(SETTINGS, STATES, captured={"a_loss_that_does_not_exist"})
