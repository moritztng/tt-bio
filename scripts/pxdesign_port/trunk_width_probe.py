"""Run tt-bio's Protenix `Trunk` at a PXDesign-pinned width (c_z=128) and report which
L1/grid gates it lands on, against the shipped c_z=256 at the same token counts.

The port's remaining structural unknown: every L1 and grid constant in `tt_bio.tenstorrent`
was fitted at c_z=256 (Protenix-v2) or c_z=384 (OpenDDE). Halving the pair channel halves
every pair footprint, so a gate can open or close silently. This runs the real class on the
real weights rather than reasoning about the constants.

Every timing is the median of `--reps` WARM cycles after one discarded warm-up, and the two
widths are run interleaved per N, so neither kernel-cache warm-up nor ordering can be mistaken
for a width effect.

Usage:  TT_VISIBLE_DEVICES=1 python3 scripts/pxdesign_port/trunk_width_probe.py [--reps R] [N ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import torch

CKPT = {
    128: "/home/ttuser/pxdesign_release_data/checkpoint/protenix_base_default_v0.5.0.pt",
    256: "/home/ttuser/pxdesign_release_data/ttbio_ptx/protenix-v2.pt",
}


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def synth(N, depth=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return (
        {
            "asym_id": torch.zeros(N),
            "msa": torch.randint(0, 32, (depth, N), generator=g).float(),
            "has_deletion": torch.zeros(depth, N),
            "deletion_value": torch.zeros(depth, N),
        },
        r(N, 449),
        r(N, N, 139),
        torch.zeros(N, N),
    )


def gate_snapshot(T, N, c_z):
    """Every width- or size-conditioned choice the trunk's pair path makes at this shape."""
    return {
        "trimul_chunk": T._trimul_chunk_size(N, c_z, 1),
        "trimul_l1_max_seq": T._trimul_l1_max_seq(),
        "trimul_l1_resident": N <= T._trimul_l1_max_seq(),
        "trimul_inproj_group": T._trimul_inproj_group(N, T._trimul_chunk_size(N, c_z, 1), 1,
                                                      c_z // T._trimul_chunk_size(N, c_z, 1)),
        "n_tri_heads": c_z // 32,
        "grid": tuple(T.COMPUTE_GRID_MAIN),
    }


def main():
    argv = list(sys.argv[1:])
    reps = 3
    if "--reps" in argv:
        i = argv.index("--reps")
        reps = int(argv[i + 1]); del argv[i:i + 2]
    Ns = [int(a) for a in argv] or [128, 256, 384]
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.protenix import Trunk

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    trunks = {}
    for c_z, path in CKPT.items():
        sd = load(path)
        derived = sd["layernorm_z_cycle.weight"].shape[0]
        assert derived == c_z, f"{path}: layernorm_z_cycle says c_z={derived}, expected {c_z}"
        trunks[c_z] = Trunk(sd, T.trunk_compute_kernel_config(ckc), c_z=c_z)
        del sd

    rows = []
    for N in Ns:                       # interleaved: the two widths share every warm-up state
        for c_z, trunk in trunks.items():
            feat, s_inputs, relp, token_bonds = synth(N)
            gates = gate_snapshot(T, N, c_z)
            times, shapes, err = [], None, None
            for r in range(reps + 1):  # rep 0 is the discarded warm-up (kernel JIT)
                t0 = time.time()
                try:
                    s, z = trunk(feat, s_inputs, relp, token_bonds, n_cycles=1)
                    shapes = [tuple(s.shape), tuple(z.shape)]
                    ttnn.deallocate(s); ttnn.deallocate(z)
                except Exception as e:          # a gate that closes shows up here
                    err = f"{type(e).__name__}: {e}"[:400]
                    break
                times.append(round(time.time() - t0, 3))
            warm = sorted(times[1:])
            row = dict(c_z=c_z, N=N, cold=times[0] if times else None,
                       warm_median=warm[len(warm) // 2] if warm else None, warm=warm,
                       shapes=shapes, error=err, gates=gates,
                       clashes={str(k): v for k, v in T._TRIMUL_CHUNK_CLASH.items()},
                       dram_shapes=sorted(str(x) for x in T._TRIMUL_DRAM_SHAPES))
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = os.path.join(os.path.dirname(__file__), "trunk_width_probe.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
