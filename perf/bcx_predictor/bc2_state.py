"""The PD-L1 design state, built by BindCraft 2 rather than transcribed.

`examples/pdl1.json` through BC2s own settings loader, then
`initialize_design_trajectory`, gives the `(protein_states, losses)` pair that
`trajectory.py:131` hands to the predictor. Everything downstream of this module --
the reference arm and the tt-bio arm -- is graded on the same pair.
"""
import os
import sys

BC2 = os.environ.get("BCX_BC2", "/home/ttuser/bcx_e2e/bc2")
if BC2 not in sys.path:
    sys.path.insert(0, BC2)

import jax                                                             # noqa: E402

from bindcraft.preflight import cleaned_campaign_settings               # noqa: E402
from bindcraft.settings import (build_design_settings, design_stage_rounds,   # noqa: E402
                                gradient_stage_rounds, parse_setting_overrides,
                                read_settings, select_design_and_validation_models)
from bindcraft.trajectory import initialize_design_trajectory           # noqa: E402
from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL                   # noqa: E402


def campaign_settings(path=None, overrides=()):
    path = path or os.path.join(BC2, "examples", "pdl1.json")
    return cleaned_campaign_settings(read_settings(path, parse_setting_overrides(overrides)))


def design_state(settings):
    """`(design_settings, protein_states, losses)` at the trajectorys first step."""
    design_settings = build_design_settings(settings)
    protein_states, _multi_chain_binders, losses = initialize_design_trajectory(
        design_settings, jax.random.PRNGKey(design_settings.seed))
    return design_settings, protein_states, losses


def state_shape(protein_states):
    return {state: {chain: len(protein) for chain, protein in complex_.items()}
            for state, complex_ in protein_states.items()}


def stage_plan(settings):
    """BindCraft 2s own step budget. Gradient stages are what the predictor pays for."""
    rounds = design_stage_rounds(settings)
    gradient = gradient_stage_rounds(settings)
    return {"per_stage": rounds, "gradient_steps": sum(gradient.values()),
            "mutate_steps": rounds["mutate"]}
