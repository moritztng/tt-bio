#!/usr/bin/env python3
"""Highest-fidelity cheap reproducer for the protenix-v1 >=512 aa wedge.

The wedge is `_diffusion_pair_cond`'s `linear_no_bias_z` matmul (tt_bio/protenix.py:2081).
Driving that method alone -- with the real weights, the real captured z_trunk/relp, the real
FLOAT32 dtype and compute_kernel_config, and with DRAM both occupied and fragmented -- does NOT
reproduce it (see perf/pxv1/hang_op_512_pc0.txt, four negative attempts). The one ingredient
none of those supplied is the TRUNK having actually executed in the same process first.

So this loops `_trunk_cond`, which is trunk + pair-cond and everything a fold does before the
sampler. Two reasons it is the right shape:
  * it includes the trunk, so the allocator/device state the pair-cond inherits is authentic;
  * it stays in ONE process across iterations, so a trial costs a trunk (~8 s) instead of a
    process start + checkpoint load + device open (~25 s), which matters when the failure rate
    is well under one.

Each iteration prints its wall time, so a wedge is a line that never arrives. Attach py-spy
from outside on the pid holding /dev/tenstorrent fds.

    TT_VISIBLE_DEVICES=0 ... python3 scripts/protenix_v1_port/repro_trunk_paircond_loop.py \
        perf/size512/fixtures/cdk2x2_512.yaml [iters]
"""
import sys
import time
from pathlib import Path

import torch

from tt_bio import weights
from tt_bio.main import _read_bio_chains, _read_bio_constraints
from tt_bio.protenix import Protenix
from tt_bio.protenix_data import build_complex_features


def main(target, iters):
    chains = _read_bio_chains(Path(target))
    bonds = _read_bio_constraints(Path(target))
    # single-sequence: matches the ladder config the wedge was characterised at
    specs = [(seq, None, mt) for _cid, seq, _spec, mt in chains]
    ids = [cid for cid, _s, _sp, _mt in chains]
    feats = build_complex_features(specs, mol_dir=str(weights.fetch("mols")),
                                   chain_ids=ids, bonds=bonds)
    n_tok = feats["residue_index"].shape[-1]
    print(f"target={target} N_token={n_tok} N_atom={feats['ref_pos'].shape[0]}", flush=True)

    ckpt = weights.fetch("protenix-v1")
    t0 = time.time()
    model = Protenix.load_from_checkpoint(str(ckpt))
    print(f"loaded in {time.time() - t0:.1f}s  diffusion dtype={model.diffusion.dtype} "
          f"c_z={model.trunk.C_Z} n_cycles={model.trunk.N_CYCLES}", flush=True)

    # MODE=full loops the whole fold() instead of just _trunk_cond, which adds the
    # diffusion/confidence phase. That is the one difference between this in-process loop
    # (12/12 clean) and the worker path (hangs on fold 3 of a 6-copy directory) that is NOT
    # the worker/spawn context, so running both modes separates "diffusion leaves state
    # behind" from "something about the worker process".
    import os
    full = os.environ.get("REPRO_MODE", "trunk") == "full"
    print(f"mode={'full fold()' if full else 'trunk_cond only'}", flush=True)

    hung = None
    for i in range(iters):
        t0 = time.time()
        try:
            if full:
                coords = model.fold(feats, n_step=6, n_sample=1, seed=0)
                dt = time.time() - t0
                fp = float(torch.as_tensor(coords).float().abs().max())
                print(f"iter {i:3d}  {dt:7.2f}s  coords_absmax={fp:.4f}", flush=True)
                del coords
                continue
            cond, aux = model._trunk_cond(feats)
        except Exception as e:                      # noqa: BLE001 - report, keep looping
            print(f"iter {i:3d}  RAISED {type(e).__name__}: {e}", flush=True)
            continue
        dt = time.time() - t0
        pz = cond["pair_z"]
        fp = float(torch.as_tensor(pz).float().abs().max())
        print(f"iter {i:3d}  {dt:7.2f}s  pair_z_absmax={fp:.4f}", flush=True)
        del cond, aux
    print(f"TRUNK+PAIRCOND LOOP COMPLETED  iters={iters}  hung={hung}", flush=True)
    return 0


if __name__ == "__main__":
    tgt = sys.argv[1] if len(sys.argv) > 1 else "perf/size512/fixtures/cdk2x2_512.yaml"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sys.exit(main(tgt, n))
