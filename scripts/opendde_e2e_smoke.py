"""End-to-end OpenDDE co-fold smoke test.

Folds examples/prot.yaml's sequence (PDB 7ROA, 117 residues -- the same target
scripts/release_gate.py uses for Protenix-v2/Boltz-2/ESMFold2) with REDUCED settings
(n_cycles, n_step) to check the full residue-trunk -> structural-expand -> structural-
diffusion path runs to completion and produces a finite, structurally plausible output.
This is NOT a production accuracy read (release_gate.py's is 10 cycles / 200 steps / 5
samples) -- it is the first real coordinate output from the wired pipeline.

OPENDDE_NCYCLES / OPENDDE_NSTEP / OPENDDE_SEED env vars override the defaults (2 cycles,
20 steps, seed 0) for a production-setting or multi-seed run; output goes to
/tmp/opendde_e2e_coords_seed<SEED>.pt so multiple seeds don't clobber each other.

Run: TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<...> PYTHONPATH=<worktree> \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/opendde_e2e_smoke.py
"""
import os
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
import sys
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
from tt_bio.protenix_data import build_complex_features

torch.set_grad_enabled(False)

SEQ = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
       "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")


def main():
    t0 = time.time()
    seed = int(os.environ.get("OPENDDE_SEED", "0"))
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_opendde_checkpoint()
    model = OpenDDE(sd, ckc, dev)
    print(f"[{time.time()-t0:.1f}s] model built", flush=True)

    feats = build_complex_features([(SEQ, None, "protein")])
    print(f"[{time.time()-t0:.1f}s] features built: N_atom={feats['ref_pos'].shape[0]} "
          f"N_res={feats['restype'].shape[0]}", flush=True)

    # OPENDDE_TRACE=1 threads fold(trace=True) -- replays a captured ttnn trace
    # of the shared denoise stream (lossless; see perf/opendde_trace_step_parity/).
    trace = os.environ.get("OPENDDE_TRACE", "0") in ("1", "true", "True")
    coords = model.fold(feats, n_step=int(os.environ.get("OPENDDE_NSTEP", "20")),
                         n_cycles=int(os.environ.get("OPENDDE_NCYCLES", "2")), seed=seed,
                         trace=trace)
    print(f"[{time.time()-t0:.1f}s] fold() returned {tuple(coords.shape)} "
          f"finite={torch.isfinite(coords).all().item()}", flush=True)
    print("coords mean/std:", coords.mean().item(), coords.std().item())
    out = f"/tmp/opendde_e2e_coords_seed{seed}.pt"
    torch.save(coords, out)
    torch.save(coords, "/tmp/opendde_e2e_coords.pt")  # back-compat: last-run convenience copy
    print(f"saved {out}")
    print("RESULT: PASS (finite coords produced)" if torch.isfinite(coords).all() else "RESULT: FAIL (non-finite)")


if __name__ == "__main__":
    main()
