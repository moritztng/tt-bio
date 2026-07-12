"""Confidence-head + best-of-N verification for OpenDDE (docs/opendde-port.md item 2).

Checks that OpenDDE.fold(n_sample=N, return_confidence=True) runs finite end-to-end and
that the confidence dict (per-sample pLDDT/pTM) is sane (0-1 range for pLDDT, populated
PAE/PDE) -- the residue-axis ConfidenceHead call (select_pair_output_branch(pair_output_
space="residue")) reused verbatim from tt_bio.protenix.Protenix, now c_z-parametrized
(zf.shape[-1] instead of hardcoded 256) so it runs at OpenDDE's c_z=384 without a
LayerNorm shape-mismatch crash.

Run: TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<...> PYTHONPATH=<worktree> \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/opendde_confidence_verify.py
"""
import os
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
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
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_opendde_checkpoint()
    model = OpenDDE(sd, ckc, dev)
    print(f"[{time.time()-t0:.1f}s] model built", flush=True)

    feats = build_complex_features([(SEQ, None, "protein")])
    n_cycles = int(os.environ.get("OPENDDE_NCYCLES", "10"))
    n_step = int(os.environ.get("OPENDDE_NSTEP", "200"))
    n_sample = int(os.environ.get("OPENDDE_NSAMPLE", "5"))
    coords, confs = model.fold(feats, n_step=n_step, n_cycles=n_cycles, seed=0,
                                n_sample=n_sample, return_confidence=True)
    print(f"[{time.time()-t0:.1f}s] fold() returned {tuple(coords.shape)} "
          f"finite={torch.isfinite(coords).all().item()}", flush=True)

    def score(c):
        ptm, iptm = c.get("ptm", 0.0), c.get("iptm", 0.0)
        return (0.8 * iptm + 0.2 * ptm) if iptm > 0.0 else (ptm if ptm > 0.0 else c["plddt"])

    ok = True
    for k, c in enumerate(confs):
        plddt, ptm = c["plddt"], c.get("ptm", 0.0)
        sane = torch.isfinite(c["pae"]).all() and torch.isfinite(c["pde"]).all() and 0.0 <= plddt <= 1.0
        ok = ok and bool(sane)
        print(f"  sample {k}: plddt={plddt:.4f} ptm={ptm:.4f} iptm={c.get('iptm', 0.0):.4f} "
              f"score={score(c):.4f} sane={bool(sane)}")

    order = sorted(range(len(confs)), key=lambda k: score(confs[k]), reverse=True)
    print(f"best-of-{n_sample} pick: sample {order[0]} (score {score(confs[order[0]]):.4f})")
    best_coords = coords[order[0]]
    torch.save(best_coords.unsqueeze(0), "/tmp/opendde_e2e_coords_bestofn.pt")
    print("RESULT: PASS" if ok and torch.isfinite(coords).all() else "RESULT: FAIL")


if __name__ == "__main__":
    main()
