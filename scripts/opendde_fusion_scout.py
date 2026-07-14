"""OpenDDE full-forward-path fusion scout.

Profiles every OpenDDE-specific surface of OpenDDE.fold on a real 7ROA predict
(card 2, real opendde.pt weights, warm program cache, device-synchronized):

  - input embedder + trunk      (SHARED protenix, residue axis)        -- closed refs
  - expand_and_refine           (expander [closed] + 4-block refiner)  -- closed refs
  - conditioning glue           (OpenDDE-specific, once per fold):
        * relp_struct             = _generate_relp on structural-token axis (host)
        * diffusion_pair_cond     = _diffusion_pair_cond(z_st, relp_struct)  [shared method,
                                    OpenDDE adds the z_trunk 384->128 compress branch]
        * plm_z_term              = _plm_z_term(pair_z, a2s, ...) host scatter
        * dit_block_biases        = _dit_block_biases(dit_z, structural_pair_attn_bias)
                                    [the 24 ttnn.add that inject the expander bias]
  - diffusion sampler           (SHARED DiT on structural-token axis)  -- closed refs

Reports device-synced wall time per piece + share of fold, at production settings
(10 cycles / 200 steps) so the Amdahl ceiling is honest. Also counts ttnn ops issued
inside the OpenDDE-specific glue (the only place a device fusion could land).

Run: TT_VISIBLE_DEVICES=2 TT_MESH_GRAPH_DESC_PATH=<p150>.textproto \
     PYTHONPATH=<worktree> /home/ttuser/tt-bio-dev/env/bin/python3 \
     scripts/opendde_fusion_scout.py
"""
import os
os.environ.setdefault("TT_VISIBLE_DEVICES", "2")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
import json
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
from tt_bio.protenix_data import build_complex_features
from tt_bio.opendde_data import build_structural_token_features
from tt_bio import protenix as _P

torch.set_grad_enabled(False)

SEQ_7ROA = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
            "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")


def _timed(dev, fn, *a, **k):
    ttnn.synchronize_device(dev)
    s = time.perf_counter()
    r = fn(*a, **k)
    ttnn.synchronize_device(dev)
    return time.perf_counter() - s, r


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_opendde_checkpoint()
    model = OpenDDE(sd, ckc, dev)
    P = model._protenix

    feats = build_complex_features([(SEQ_7ROA, None, "protein")])
    ifd = build_structural_token_features(feats)
    Ns = ifd["parent_residue_idx"].shape[0]
    n_res = feats["restype"].shape[0]
    print(f"Ns={Ns} n_res={n_res} N_atom={feats['ref_pos'].shape[0]}", flush=True)

    T = {"trunk": 0.0, "expand_and_refine": 0.0, "expander_only": 0.0,
         "diffusion_sample": 0.0,
         # OpenDDE-specific conditioning glue (once per fold):
         "relp_struct": 0.0, "diffusion_pair_cond": 0.0, "plm_z_term": 0.0,
         "dit_block_biases": 0.0, "generate_relp_total": 0.0, "generate_relp_calls": 0}
    # ttnn op count inside the OpenDDE-specific glue
    COUNT = {"linear": 0, "layer_norm": 0, "add": 0, "concat": 0,
             "reshape": 0, "matmul": 0, "other": 0}
    _counting = {"on": False}

    def _wrap_op(name, orig):
        def w(*a, **k):
            if _counting["on"]:
                COUNT[name if name in COUNT else "other"] = COUNT.get(name, 0) + 1
            return orig(*a, **k)
        return w

    for nm in ("linear", "layer_norm", "add", "concat", "reshape", "matmul"):
        setattr(ttnn, nm, _wrap_op(nm, getattr(ttnn, nm)))

    # ---- wrap the OpenDDE-specific glue methods ----
    orig_dpc = P._diffusion_pair_cond
    def timed_dpc(*a, **k):
        _counting["on"] = True
        t, r = _timed(dev, orig_dpc, *a, **k)
        _counting["on"] = False
        T["diffusion_pair_cond"] += t
        return r
    P._diffusion_pair_cond = timed_dpc

    orig_plm = P._plm_z_term
    def timed_plm(*a, **k):
        t0 = time.perf_counter()
        r = orig_plm(*a, **k)
        T["plm_z_term"] += time.perf_counter() - t0
        return r
    P._plm_z_term = timed_plm

    orig_relp = P._generate_relp
    def timed_relp(*a, **k):
        t0 = time.perf_counter()
        r = orig_relp(*a, **k)
        T["generate_relp_total"] += time.perf_counter() - t0
        T["generate_relp_calls"] += 1
        return r
    P._generate_relp = timed_relp

    # dit_block_biases / dit_pair_biases carry the structural_pair_attn_bias injection
    orig_dbb = P.diffusion._dit_block_biases
    def timed_dbb(*a, **k):
        _counting["on"] = True
        t, r = _timed(dev, orig_dbb, *a, **k)
        _counting["on"] = False
        T["dit_block_biases"] += t
        return r
    P.diffusion._dit_block_biases = timed_dbb
    orig_dpb = P.diffusion._dit_pair_biases
    def timed_dpb(*a, **k):
        t0 = time.perf_counter()
        r = orig_dpb(*a, **k)
        T["dit_block_biases"] += time.perf_counter() - t0
        return r
    P.diffusion._dit_pair_biases = timed_dpb

    # ---- wrap major phases (share-script style) ----
    orig_expand = model.expand_and_refine
    def timed_expand(*a, **k):
        t, r = _timed(dev, orig_expand, *a, **k)
        T["expand_and_refine"] += t
        return r
    model.expand_and_refine = timed_expand

    orig_expander = model.expander
    class _ExpProxy:
        def __init__(self, orig): self._orig = orig
        def __call__(self, *a, **k):
            t, r = _timed(dev, orig_expander, *a, **k)
            T["expander_only"] += t
            return r
        def __getattr__(self, n): return getattr(orig_expander, n)
    model.expander = _ExpProxy(orig_expander)

    orig_trunk = P.trunk
    class _CallProxy:
        def __init__(self, orig, fn): self._orig = orig; self._fn = fn
        def __call__(self, *a, **k):
            t, r = _timed(dev, self._fn, *a, **k)
            T["trunk"] += t
            return r
        def __getattr__(self, n): return getattr(self._orig, n)
    P.trunk = _CallProxy(orig_trunk, orig_trunk)

    orig_sample = _P.edm_sample
    def timed_sample(diffusion, cond, N, n_step=20, seed=None):
        t, r = _timed(dev, orig_sample, diffusion, cond, N, n_step=n_step, seed=seed)
        T["diffusion_sample"] += t
        return r
    _P.edm_sample = timed_sample

    n_step = int(os.environ.get("OPENDDE_NSTEP", "200"))
    n_cycles = int(os.environ.get("OPENDDE_NCYCLES", "10"))
    seed = int(os.environ.get("OPENDDE_SEED", "0"))

    print("warming...", flush=True)
    t0 = time.time()
    model.fold(feats, n_step=2, n_cycles=1, seed=seed)
    print(f"warm done in {time.time()-t0:.1f}s", flush=True)

    # reset accumulators (warm polluted them)
    for k in T: T[k] = 0.0 if isinstance(T[k], float) else 0
    for k in COUNT: COUNT[k] = 0

    ttnn.synchronize_device(dev)
    s0 = time.perf_counter()
    coords = model.fold(feats, n_step=n_step, n_cycles=n_cycles, seed=seed)
    ttnn.synchronize_device(dev)
    total = time.perf_counter() - s0

    glue = (T["diffusion_pair_cond"] + T["plm_z_term"] + T["dit_block_biases"] +
            T["generate_relp_total"])
    accounted = (T["trunk"] + T["expand_and_refine"] + T["diffusion_sample"] + glue)
    rec = {
        "target": "7ROA", "Ns": Ns, "n_res": n_res, "n_cycles": n_cycles,
        "n_step": n_step, "seed": seed, "finite": bool(torch.isfinite(coords).all().item()),
        "total_fold_s": round(total, 4),
        "trunk_s": round(T["trunk"], 4), "trunk_share": round(T["trunk"]/total, 5),
        "expand_and_refine_s": round(T["expand_and_refine"], 4),
        "expander_only_s": round(T["expander_only"], 4),
        "expander_only_share": round(T["expander_only"]/total, 5),
        "refiner_seam_s": round(T["expand_and_refine"]-T["expander_only"], 4),
        "diffusion_sample_s": round(T["diffusion_sample"], 4),
        "diffusion_share": round(T["diffusion_sample"]/total, 5),
        "OPENDDE_GLUE": {
            "relp_struct+residue_relp_s": round(T["generate_relp_total"], 4),
            "relp_calls": T["generate_relp_calls"],
            "diffusion_pair_cond_s": round(T["diffusion_pair_cond"], 4),
            "plm_z_term_s": round(T["plm_z_term"], 4),
            "dit_block_biases_s": round(T["dit_block_biases"], 4),
            "glue_total_s": round(glue, 4),
            "glue_share": round(glue/total, 5),
        },
        "GLUE_TTNN_OPS_on_device": dict(COUNT),
        "residual_gap_s": round(total - accounted, 4),
        "residual_gap_share": round((total - accounted)/total, 5),
        "amdahl_ceiling_opendde_specific": round(1.0/(1.0 - (T["expander_only"]+glue)/total), 4),
    }
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
