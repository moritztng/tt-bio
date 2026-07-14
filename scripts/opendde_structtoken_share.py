"""Measure StructuralTokenExpander's real share of a full OpenDDE fold.

Real opendde.pt weights, real 7ROA input, warm program cache, device-synchronized timings.
The expander runs ONCE per fold (not per recycle/diffusion step), so its share shrinks as
n_cycles/n_step grow -- this reports both a reduced-setting and a production-setting share
so the Amdahl ceiling is honest.

OPENDDE_NCYCLES / OPENDDE_NSTEP / OPENDDE_SEED env vars override (defaults 2 cycles, 20 steps,
seed 0). Prints one JSON line per fold with: total fold s, expand_and_refine s (+ its share),
trunk s, diffusion s, Ns, n_res.

Run: PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=0 \
    /home/moritz/tt-bio/env/bin/python3 scripts/opendde_structtoken_share.py
"""
import os
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
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

    feats = build_complex_features([(SEQ_7ROA, None, "protein")])
    ifd = build_structural_token_features(feats)
    Ns = ifd["parent_residue_idx"].shape[0]
    n_res = feats["restype"].shape[0]
    print(f"Ns={Ns} n_res={n_res} N_atom={feats['ref_pos'].shape[0]}", flush=True)

    # Instrument: wrap expand_and_refine, trunk, edm_sample to time each phase.
    times = {"expand_and_refine": 0.0, "expander_only": 0.0, "trunk": 0.0, "diffusion_sample": 0.0}
    orig_expand = model.expand_and_refine

    def timed_expand(*a, **k):
        t, r = _timed(dev, orig_expand, *a, **k)
        times["expand_and_refine"] += t
        return r
    model.expand_and_refine = timed_expand

    # StructuralTokenExpander alone (the novel block), inside expand_and_refine.
    orig_expander = model.expander

    class _ExpProxy:
        def __init__(self, orig):
            self._orig = orig

        def __call__(self, *a, **k):
            t, r = _timed(dev, orig_expander, *a, **k)
            times["expander_only"] += t
            return r

        def __getattr__(self, name):
            return getattr(orig_expander, name)
    model.expander = _ExpProxy(orig_expander)

    orig_trunk = model._protenix.trunk

    class _CallProxy:
        """Wrap a callable module so __call__ is timed but all other attrs (e.g. C_Z) delegate."""
        def __init__(self, orig, fn):
            self._orig = orig
            self._fn = fn

        def __call__(self, *a, **k):
            t, r = _timed(dev, self._fn, *a, **k)
            times["trunk"] += t
            return r

        def __getattr__(self, name):
            return getattr(self._orig, name)
    model._protenix.trunk = _CallProxy(orig_trunk, orig_trunk)

    orig_sample = _P.edm_sample

    def timed_sample(diffusion, cond, N, n_step=20, seed=None):
        t, r = _timed(dev, orig_sample, diffusion, cond, N, n_step=n_step, seed=seed)
        times["diffusion_sample"] += t
        return r
    _P.edm_sample = timed_sample

    n_step = int(os.environ.get("OPENDDE_NSTEP", "20"))
    n_cycles = int(os.environ.get("OPENDDE_NCYCLES", "2"))
    seed = int(os.environ.get("OPENDDE_SEED", "0"))

    # warm a reduced fold first (program cache) -- tiny settings, discard result
    print("warming...", flush=True)
    t0 = time.time()
    try:
        model.fold(feats, n_step=2, n_cycles=1, seed=seed)
    except Exception as e:
        print("warm fold failed:", e, flush=True)
    print(f"warm done in {time.time()-t0:.1f}s", flush=True)

    times["expand_and_refine"] = 0.0
    times["expander_only"] = 0.0
    times["trunk"] = 0.0
    times["diffusion_sample"] = 0.0
    t0 = time.perf_counter()
    ttnn.synchronize_device(dev)
    coords = model.fold(feats, n_step=n_step, n_cycles=n_cycles, seed=seed)
    ttnn.synchronize_device(dev)
    total = time.perf_counter() - t0

    rec = {
        "target": "7ROA", "Ns": Ns, "n_res": n_res,
        "n_cycles": n_cycles, "n_step": n_step, "seed": seed,
        "total_fold_s": round(total, 3),
        "expand_and_refine_s": round(times["expand_and_refine"], 4),
        "expander_refiner_share": round(times["expand_and_refine"] / total, 5),
        "expander_only_s": round(times["expander_only"], 4),
        "expander_only_share": round(times["expander_only"] / total, 5),
        "trunk_s": round(times["trunk"], 4),
        "trunk_share": round(times["trunk"] / total, 5),
        "diffusion_sample_s": round(times["diffusion_sample"], 4),
        "diffusion_share": round(times["diffusion_sample"] / total, 5),
        "finite": bool(torch.isfinite(coords).all().item()),
    }
    print(json.dumps(rec, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
