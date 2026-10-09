"""The one object a diffusion sampler sees: OpenDDE 1.2.0's constraint guidance.

``tt_bio.protenix.edm_sample(..., guidance=g)`` calls ``g.step`` in place of its Euler
update. That is the whole interface, so any model whose sampler is ``edm_sample`` (Protenix-v2,
OpenDDE) or calls the same method can be guided; OpenDDE wires it first.

``step`` reproduces upstream's ``sample_diffusion`` with ``guidance.enable`` (opendde/model/
generator.py, v1.2.0):

1. ``TFGEngine.update`` on the denoiser's clean estimate x0: constraint projection, the rigid
   early pass (``epitope.guide_x0``, steps 100, 103, ..., 187 by default), then 20 gradient
   steps on the physics potentials, then the predictor-corrector update from the corrected x0.
2. The rigid late pass on the sampled state: a coarse pose search at step 190 and at the last
   step, a refinement at every even step from 190 and at the last step (120 iterations at the
   last step, else 40).

Rigid guidance runs when the input carries contact pairs it can apply (``movable_chains``, or
a two-chain complex) or an epitope; it then switches the atom-level contact restraint off,
as upstream does. Without a constraint, guidance is the physics potentials alone, which is what
upstream's ``--use_tfg_guidance true`` does to an unconstrained input.
"""
from __future__ import annotations

from . import epitope, rigid
from .engine import TFGEngine, default_guidance_config, parse_tfg_config
from .rigid import RigidSchedule


class Guidance:
    """Constraint guidance for one target.

    feats: the model's atom-level feature dict plus the guidance features
    (``tt_bio.tfg.features``): restraint indices, the movable-chain mask, the epitope
    request and the geometry tables the potentials read. All on the host, atom order equal
    to the sampler's coordinates.
    """

    def __init__(self, feats, *, schedule: RigidSchedule = RigidSchedule(), config=None):
        cfg = dict(default_guidance_config() if config is None else config)
        cfg["enable"] = True
        self.cfg = parse_tfg_config(cfg)
        self.feats = feats
        self.schedule = schedule
        self.mode = epitope.rigid_mode(feats, True, schedule)
        if self.mode != "on" and epitope.active(feats):
            raise ValueError(
                "constraint.epitope is applied only by rigid-body guidance; "
                "do not set the rigid mode to off or control, or remove constraint.epitope.")
        if self.mode != "off":
            for term in self.cfg.terms:
                if term.name == "UserDistanceRestraintPotential":
                    term.interval = 0
                    term.enable_projection = False
        if self.mode == "on":
            pairs = feats.get("user_distance_restraint_index")
            if not epitope.active(feats) and (pairs is None or pairs.numel() == 0):
                raise ValueError(
                    "rigid mode on needs either contact pairs or an epitope request; "
                    "there is nothing to apply.")
            # A request that cannot be applied fails here, not ~100 steps into the sampler.
            import torch
            probe = torch.empty(1, feats["atom_to_token_idx"].shape[-1], 3)
            (epitope.groups if epitope.active(feats) else rigid.contact_groups)(probe, feats)
        self.engine = TFGEngine(self.cfg)

    def _x0_hook(self, x0, step_i):
        return epitope.guide_x0(x0, self.feats, step_i, self.schedule)

    def step(self, x_noisy, x0, *, t_hat, sigma_t, eta, step, n_step):
        """One sampler step from x_noisy at t_hat to the next state at sigma_t, given the
        denoiser's x0. Shapes [M, N_atom, 3], fp32 host tensors."""
        x = self.engine.update(
            x_noisy, x0, t_hat=t_hat, c_tau=sigma_t, step_scale_eta=eta, step_i=step,
            num_diffusion_steps=n_step, feats=self.feats, x0_hook=self._x0_hook)
        if self.mode != "on" or not rigid.late_pass_enabled(self.schedule):
            return x
        coarse, refine = rigid.intervention_schedule(n_step, self.schedule)
        pocket = epitope.active(self.feats)
        if step in coarse:
            x = (epitope.search_epitope if pocket else rigid.search_rigid_contact)(x, self.feats)
        if step in refine:
            iterations = 120 if step == n_step - 1 else 40
            x = (epitope.refine_epitope if pocket else rigid.refine_rigid_contact)(
                x, self.feats, iterations=iterations)
        return x
