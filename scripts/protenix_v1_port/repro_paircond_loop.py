#!/usr/bin/env python3
"""Tight-loop repro for the protenix-v1 512 aa wedge, driving the REAL pair-cond code path.

The wedge is the `linear_no_bias_z` matmul in tt_bio/protenix.py::_diffusion_pair_cond.
An earlier bare-ttnn repro of that matmul alone did NOT reproduce it -- but it used bfloat16,
and the real op runs in FLOAT32 (the diffusion pair branch is fp32 by default:
PROTENIX_DIFFUSION_FP32_DEVICE, protenix.py:885). Rather than guess which of dtype /
compute_kernel_config / input provenance matters, this calls the real method with the real
weights and the real z_trunk+relp captured from a fold, so every one of those is authentic.

Capture the inputs first with one instrumented fold:

    TT_PROTENIX_PAIRCOND_DEBUG=1 TT_PROTENIX_PAIRCOND_DEBUG_DIR=<dir> \
      python3 -m tt_bio.main predict perf/size512/fixtures/cdk2x2_512.yaml \
      --model protenix-v1 --single_sequence --sampling_steps 6 --diffusion_samples 1

then:  python3 scripts/protenix_v1_port/repro_paircond_loop.py <dir>/paircond_inputs.pt [iters]

Each iteration syncs and prints its wall time, so a wedge shows as a call that never returns
rather than as a slow fold.
"""
import sys
import time

import torch
import ttnn

from tt_bio import weights
from tt_bio.protenix import Protenix


def main(inputs_path, iters):
    blob = torch.load(inputs_path, map_location="cpu", weights_only=False)
    z_trunk, relp = blob["z_trunk"], blob["relp"]
    print(f"z_trunk {tuple(z_trunk.shape)} {z_trunk.dtype}   "
          f"relp {tuple(relp.shape)} {relp.dtype}   captured dtype={blob.get('dtype')}",
          flush=True)

    ckpt = weights.fetch("protenix-v1")
    print(f"loading {ckpt}", flush=True)
    t0 = time.time()
    model = Protenix.load_from_checkpoint(str(ckpt))
    print(f"loaded in {time.time() - t0:.1f}s   diffusion dtype={model.diffusion.dtype} "
          f"fp32={model.diffusion._diffusion_fp32}", flush=True)

    # Optional allocator CHURN before the loop. The pair-cond alone does not wedge even with
    # real weights/inputs/dtypes, and the pcdbg fold shows the queue was DRAINED right before
    # linear_z (its own _sync ran), so a deep async queue is not required either. The remaining
    # difference from this repro is that in a real fold the trunk has just run: ~200 ops of
    # alloc/free at many sizes, leaving a fragmented DRAM arena with its tensors still live.
    # An earlier test used 6 clean 256 MiB blocks -- occupancy without fragmentation. This
    # churns mixed sizes and frees a random half, which is what fragmentation actually is.
    import os
    churn_mb = int(os.environ.get("REPRO_CHURN_MB", "0"))
    _keep = []
    if churn_mb:
        import random
        random.seed(0)
        made = 0
        while made < churn_mb:
            mb = random.choice([8, 16, 32, 64, 128])
            rows = mb * 2 ** 20 // 2 // 1024
            try:
                _keep.append(ttnn.from_torch(torch.zeros(1, rows, 1024),
                                             layout=ttnn.TILE_LAYOUT, device=model.dev,
                                             dtype=ttnn.bfloat16))
            except Exception as e:
                print(f"churn stopped at {made} MiB: {e}", flush=True)
                break
            made += mb
        # free a random half, leaving holes rather than a clean high-water mark
        random.shuffle(_keep)
        for t_ in _keep[: len(_keep) // 2]:
            ttnn.deallocate(t_)
        _keep = _keep[len(_keep) // 2:]
        print(f"churn: allocated ~{made} MiB in mixed blocks, freed half -> "
              f"{len(_keep)} blocks still live", flush=True)

    slow = 0
    for i in range(iters):
        # Fresh upload each iteration: the real fold hands _diffusion_pair_cond a device
        # tensor the trunk just wrote, and re-uploading is the closest thing available
        # without replaying the whole trunk.
        z_tt = ttnn.from_torch(z_trunk, layout=ttnn.TILE_LAYOUT, device=model.dev,
                               dtype=ttnn.bfloat16)
        t0 = time.time()
        pz_host = model._diffusion_pair_cond(z_tt, relp)
        ttnn.synchronize_device(model.dev)
        dt = time.time() - t0
        if isinstance(pz_host, torch.Tensor):
            fp = float(pz_host.float().abs().max())
        else:
            fp = float("nan")
        print(f"iter {i:3d}  {dt:7.2f}s  pz_absmax={fp:.4f}", flush=True)
        if dt > 20:
            slow += 1
        try:
            ttnn.deallocate(z_tt)
        except Exception:
            pass
    print(f"LOOP COMPLETED  slow_iters(>20s)={slow}", flush=True)
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pv1/pcdbg_dir/paircond_inputs.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    sys.exit(main(p, n))
