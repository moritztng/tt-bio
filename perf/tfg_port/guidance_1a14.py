"""Integrated parity and CPU cost of tt_bio.tfg.guidance.Guidance on the 1a14 examples.

Builds the fold's features from examples/tfg/1a14_{contact,pocket}.yaml the way the worker
does (single-sequence specs; the MSA does not enter the guidance features), runs Guidance.step at a
handful of sampler steps on a synthetic structure, and runs the same step through upstream v1.2.0
(TFGEngine.step with the denoiser stubbed to return the same x0, then the late rigid pass in
generator.py's order). Prints max |ours - upstream| and wall time per step.

    PYTHONPATH=. python perf/tfg_port/guidance_1a14.py [contact|pocket] [steps...]
"""
import os
import sys
import time

import torch

UP = os.environ.get("TFG_UPSTREAM", "/tmp/tfgsrc/up")
os.environ.update(OPENDDE_VINA_FAST="off", OPENDDE_RIGID_CONTACT="auto", OPENDDE_RIGID_CORE="off")
N = 200


def feats_for(kind):
    from tt_bio.main import _read_bio_bonds, _read_bio_chains
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.tfg import features as F
    from tt_bio.worker import _tfg_guidance

    from pathlib import Path
    path = Path(f"examples/tfg/1a14_{kind}.yaml")
    chains = _read_bio_chains(path)
    bonds = _read_bio_bonds(path, chains)
    feats = build_complex_features([(s, None, mt) for _c, s, _sp, mt, _m in chains],
                                   chain_ids=[c for c, *_r in chains], bonds=bonds,
                                   modifications=[m for *_x, m in chains])
    t = time.time()
    g = _tfg_guidance(path, {"use_tfg_guidance": True}, feats, chains, bonds)
    print(f"{kind}: {feats['atom_to_token_idx'].shape[-1]} atoms, {int(feats['asym_id'].max()) + 1} chains,"
          f" mode={g.mode}, guidance build {time.time() - t:.1f} s", flush=True)
    return g


def structure(feats, seed=0):
    """Chains as bonded random walks over tokens, the antibody chains placed 25 A off the antigen,
    so the request is unmet and the search has to move them."""
    g = torch.Generator().manual_seed(seed)
    tok = feats["atom_to_token_idx"]
    n_tok = int(tok.max()) + 1
    d = torch.randn(n_tok, 3, generator=g)
    path = torch.cumsum(d / d.norm(dim=-1, keepdim=True) * 3.8, 0)
    asym = feats["asym_id"]
    path = path + 25.0 * torch.nn.functional.one_hot(asym, int(asym.max()) + 1).float()[:, :3]
    ref = feats["ref_pos"] - feats["ref_pos"].mean(0)
    return (path[tok] + 0.3 * ref).unsqueeze(0).float()


def upstream_step(mods, gf, x_noisy, x0, k):
    from opendde.tfg.config import parse_tfg_config
    from opendde.tfg.engine import TFGEngine
    up_rc, up_ep = mods
    from tt_bio.tfg.engine import default_guidance_config
    cfg = dict(default_guidance_config()); cfg["enable"] = True
    cfg = parse_tfg_config(cfg)
    for term in cfg.terms:
        if term.name == "UserDistanceRestraintPotential":
            term.interval = 0
            term.enable_projection = False
    eng = TFGEngine(cfg, device=torch.device("cpu"), dtype=torch.float32)
    x = eng.step(lambda **kw: x0, x=x_noisy, t_hat=torch.tensor(T_HAT), c_tau=torch.tensor(C_TAU),
                 step_scale_eta=ETA, step_i=k, num_diffusion_steps=N, input_feature_dict=gf,
                 s_inputs=None, s_trunk=None, z_trunk=None, pair_z=None, p_lm=None, c_l=None,
                 chunk_size=None, inplace_safe=False, enable_efficient_fusion=False,
                 torch_generator=torch.Generator().manual_seed(1))
    coarse, refine = up_rc.intervention_schedule(N)
    pocket = up_ep.active(gf)
    if k in coarse:
        x = up_ep.search_epitope(x, gf) if pocket else up_rc.search_rigid_contact(x, gf)
    if k in refine:
        it = 120 if k == N - 1 else 40
        x = up_ep.refine_epitope(x, gf, iterations=it) if pocket else up_rc.refine_rigid_contact(x, gf, iterations=it)
    return x


T_HAT, C_TAU, ETA = 1.2, 1.0, 1.5

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "contact"
    steps = [int(s) for s in sys.argv[2:]] or [0, 100, 190, 199]
    g = feats_for(kind)
    sys.path.insert(0, UP)
    from opendde.tfg import epitope_guidance as up_ep
    from opendde.tfg import rigid_contact as up_rc
    x0 = structure(g.feats)
    x_noisy = x0 + 1.2 * torch.randn(x0.shape, generator=torch.Generator().manual_seed(2))
    for k in steps:
        t = time.time()
        ours = g.step(x_noisy, x0, t_hat=torch.tensor(T_HAT), sigma_t=torch.tensor(C_TAU), eta=ETA, step=k, n_step=N)
        t_ours = time.time() - t
        t = time.time()
        ref = upstream_step((up_rc, up_ep), g.feats, x_noisy, x0, k)
        t_up = time.time() - t
        diff = (ours - ref).abs().max().item()
        moved = (ours - x_noisy).abs().max().item()
        print(f"{kind} step {k}: max|ours-upstream| {diff:.3g} A, equal={torch.equal(ours, ref)},"
              f" ours {t_ours:.1f} s, upstream {t_up:.1f} s, max move {moved:.2f} A", flush=True)
