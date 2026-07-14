"""Run a real OpenDDE fold with trace replay on so the sitecustomize parity
hook fires on the first per-step denoise. Mirrors scripts/opendde_fusion_scout.py's
7ROA setup (real opendde.pt weights, 10 cycles / 200 steps, warm)."""
import os
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
from tt_bio.protenix_data import build_complex_features

torch.set_grad_enabled(False)

SEQ_7ROA = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
            "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_opendde_checkpoint()
    model = OpenDDE(sd, ckc, dev)
    feats = build_complex_features([(SEQ_7ROA, None, "protein")])
    n_step = int(os.environ.get("OPENDDE_NSTEP", "200"))
    n_cycles = int(os.environ.get("OPENDDE_NCYCLES", "10"))
    seed = int(os.environ.get("OPENDDE_SEED", "0"))

    print("warming (trace, n_step=2)...", flush=True)
    t0 = time.time()
    model.fold(feats, n_step=2, n_cycles=1, seed=seed, trace=True)
    print(f"warm done in {time.time()-t0:.1f}s", flush=True)

    print("traced fold (n_step=%d, n_cycles=%d)..." % (n_step, n_cycles), flush=True)
    ttnn.synchronize_device(dev)
    s0 = time.perf_counter()
    coords = model.fold(feats, n_step=n_step, n_cycles=n_cycles, seed=seed, trace=True)
    ttnn.synchronize_device(dev)
    total = time.perf_counter() - s0
    print(f"traced total={total:.3f}s finite={bool(torch.isfinite(coords).all().item())}",
          flush=True)


if __name__ == "__main__":
    main()
