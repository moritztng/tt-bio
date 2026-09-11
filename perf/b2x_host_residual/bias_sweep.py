#!/usr/bin/env python3
"""Where the bias-stack win actually comes from, and how the block size behaves.

Three forms of `cat([layer(x) for layer in layers], dim=-1)`, real checkpoint weights:
  shipped   24 separate outputs, then a 384 MB cat
  preall    24 separate outputs written straight into their slice of one preallocated tensor
            (no cat, no blocking)
  blocked   preallocated output AND one pass over row blocks
Separating them says whether the lever is "stop allocating and copying 384 MB" or "keep the
block in cache", which decides whether the block size needs tuning per host at all.
"""
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = "/home/moritz/tt-bio"
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(REPO) / "scripts" / "gpu_vs_tt"))
import torch  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)
import tt_bio.boltz2 as BZ                                     # noqa: E402
from tt_bio.worker import _ensure_local_artifacts              # noqa: E402

cfg = dict(model="boltz2", msa_dir="/tmp/b2x_hostpath/msa_512",
           struct_dir="/tmp/b2x_hostpath/out")
_ensure_local_artifacts(cfg)
_diff = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003, "rho": 7,
         "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0, "P_mean": -1.2,
         "P_std": 1.5, "coordinate_augmentation": True, "alignment_reverse_diff": True,
         "synchronize_sigmas": True}
model = BZ.Boltz2.load_from_checkpoint(
    cfg["conf_ckpt"],
    predict_args={"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1,
                  "max_parallel_samples": None},
    diffusion_process_args=_diff,
    pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
    msa_args={"subsample_msa": True, "num_subsampled_msa": 1024, "use_paired_feature": True,
              "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15, "z_dropout": 0.25,
              "pairwise_head_width": 32, "pairwise_num_heads": 4,
              "activation_checkpointing": True},
    steering_args={"fk_steering": False, "physical_guidance_update": False,
                   "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                   "fk_resampling_interval": 3, "num_gd_steps": 20},
    use_kernels=False, use_tenstorrent=False, trace=False, diffusion_trace=False).eval()

layers = model.diffusion_conditioning.token_trans_proj_z
h = layers[0][1].out_features
n, c = 512, 128
g = torch.Generator().manual_seed(4242)
z = torch.randn(1, n, n, c, generator=g)
flat = z.reshape(-1, c)
M = flat.shape[0]


def tm(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), max(ts) - min(ts)


def shipped():
    return torch.cat([l(z) for l in layers], dim=-1)


def preall():
    out = flat.new_empty(M, h * len(layers))
    for i, l in enumerate(layers):
        out[:, i * h:(i + 1) * h] = l(flat)
    return out.reshape(1, n, n, h * len(layers))


def blocked(rows):
    out = flat.new_empty(M, h * len(layers))
    for s in range(0, M, rows):
        blk = flat[s:s + rows]
        for i, l in enumerate(layers):
            out[s:s + rows, i * h:(i + 1) * h] = l(blk)
    return out.reshape(1, n, n, h * len(layers))


print(f"host {os.uname().nodename}  threads {torch.get_num_threads()}  "
      f"{len(layers)} layers x {h},  z {M*c*4/2**20:.0f} MB, out {M*h*len(layers)*4/2**20:.0f} MB")
ref = shipped()
t0, sp0 = tm(shipped)
print(f"  shipped                 {1e3*t0:8.1f} ms  (spread {1e3*sp0:.1f})")
got = preall()
t1, sp1 = tm(preall)
print(f"  preallocated, no block  {1e3*t1:8.1f} ms  {t0/t1:5.2f}x  "
      f"bit-exact {torch.equal(ref, got)}  (spread {1e3*sp1:.1f})")
del got
for rows in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
    got = blocked(rows)
    t, sp = tm(lambda: blocked(rows))
    print(f"  blocked rows={rows:7d}   {1e3*t:8.1f} ms  {t0/t:5.2f}x  "
          f"bit-exact {torch.equal(ref, got)}  block {rows*(c+h*len(layers))*4/2**20:6.1f} MB  "
          f"(spread {1e3*sp:.1f})")
    del got
