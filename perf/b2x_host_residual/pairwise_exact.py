#!/usr/bin/env python3
"""Where the blocked PairwiseConditioning stops being bit-exact, with REAL checkpoint weights.

The isolated benchmark said bit-exact at every block size with randomly-initialised weights and
the real model says otherwise, so the divergence is value-dependent, not shape-dependent. Split
the stage into its three steps and find which one moves, at which block size, and by how much.
"""
import hashlib
import os
import sys
from pathlib import Path

REPO = "/home/moritz/tt-bio"
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(REPO) / "scripts" / "gpu_vs_tt"))
import torch  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)
import tt_bio.boltz2 as BZ                                    # noqa: E402
import tt_baseline as B                                       # noqa: E402
from tt_bio.main import _resolve_recycling_steps              # noqa: E402
from tt_bio.worker import _ensure_local_artifacts             # noqa: E402

B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
cfg = dict(model="boltz2", msa_dir="/tmp/b2x_hostpath/msa_512", struct_dir="/tmp/b2x_hostpath/out")
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
pc = model.diffusion_conditioning.pairwise_conditioner
n = 512
c = 128
g = torch.Generator().manual_seed(20260911)
z_trunk = torch.randn(1, n, n, c, generator=g)
relpos = torch.randn(1, n, n, c, generator=g)


def rep(tag, a, b):
    same = torch.equal(a, b)
    d = (a - b).abs()
    print(f"  {tag:34s} bit-exact {str(same):5s}  max|d| {d.max().item():.3e}  "
          f"n_diff {int((a != b).sum()):9d} / {a.numel()}")
    return same


print(f"host {os.uname().nodename}  threads {torch.get_num_threads()}  real checkpoint weights")

# step 1: cat + init proj
cat_ref = torch.cat((z_trunk, relpos), dim=-1)
ref1 = pc.dim_pairwise_init_proj(cat_ref)
zt = z_trunk.reshape(-1, c)
rp = relpos.reshape(-1, c)
for rows in (256, 512, 819, 1024, 2048, 8192, 65536):
    out = zt.new_empty(zt.shape[0], c)
    for s in range(0, zt.shape[0], rows):
        out[s:s + rows] = pc.dim_pairwise_init_proj(
            torch.cat((zt[s:s + rows], rp[s:s + rows]), dim=-1))
    rep(f"init_proj rows={rows}", ref1.reshape(-1, c), out)
    del out

# step 2: one transition on the init-proj output
tr = pc.transitions[0]
ref2 = tr(ref1) + ref1
flat1 = ref1.reshape(-1, c)
for rows in (819, 2048, 8192):
    out = flat1.new_empty(flat1.shape)
    for s in range(0, flat1.shape[0], rows):
        blk = flat1[s:s + rows]
        out[s:s + rows] = tr(blk) + blk
    rep(f"transition[0] rows={rows}", ref2.reshape(-1, c), out)
    del out

# step 3: which op inside the transition moves
x = flat1[:8192]
xr = ref1.reshape(-1, c)[:8192]
for tag, fn in (("norm", lambda t: tr.norm(t)),
                ("fc1", lambda t: tr.fc1(tr.norm(t))),
                ("fc2", lambda t: tr.fc2(tr.norm(t))),
                ("gate", lambda t: tr.silu(tr.fc1(tr.norm(t))) * tr.fc2(tr.norm(t))),
                ("fc3", lambda t: tr.fc3(tr.silu(tr.fc1(tr.norm(t))) * tr.fc2(tr.norm(t))))):
    whole = fn(ref1.reshape(-1, c))[:8192]
    part = fn(x)
    rep(f"inside transition: {tag}", whole, part)

# ---------------------------------------------------------------------------------------------
# Split the lever: how much of the win is the bit-exact part (cat + init proj) and how much
# needs the transitions, which SiLU makes inexact?
import statistics as st
import time


def tm(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def shipped():
    z = torch.cat((z_trunk, relpos), dim=-1)
    z = pc.dim_pairwise_init_proj(z)
    for t in pc.transitions:
        z = t(z) + z
    return z


def blocked_all(rows):
    out = zt.new_empty(zt.shape[0], c)
    for s in range(0, zt.shape[0], rows):
        blk = pc.dim_pairwise_init_proj(torch.cat((zt[s:s + rows], rp[s:s + rows]), dim=-1))
        for t in pc.transitions:
            blk = t(blk) + blk
        out[s:s + rows] = blk
    return out.reshape(z_trunk.shape)


def blocked_proj_only(rows):
    """The bit-exact half: block the cat + init proj, leave the transitions whole."""
    out = zt.new_empty(zt.shape[0], c)
    for s in range(0, zt.shape[0], rows):
        out[s:s + rows] = pc.dim_pairwise_init_proj(
            torch.cat((zt[s:s + rows], rp[s:s + rows]), dim=-1))
    z = out.reshape(z_trunk.shape)
    for t in pc.transitions:
        z = t(z) + z
    return z


print("\n--- how the pairwise win splits ---")
ref = shipped()
t_ship = tm(shipped)
print(f"  shipped                         {1e3*t_ship:8.1f} ms")
for rows in (819, 2048):
    got = blocked_proj_only(rows)
    t = tm(lambda: blocked_proj_only(rows))
    print(f"  proj-only blocked rows={rows:5d}    {1e3*t:8.1f} ms  {t_ship/t:5.2f}x  "
          f"bit-exact {torch.equal(ref, got)}")
    del got
for rows in (819, 2048):
    got = blocked_all(rows)
    t = tm(lambda: blocked_all(rows))
    d = (ref - got).abs()
    print(f"  full blocked      rows={rows:5d}    {1e3*t:8.1f} ms  {t_ship/t:5.2f}x  "
          f"bit-exact {torch.equal(ref, got)}  max|d| {d.max().item():.2e}  "
          f"n_diff {int((ref != got).sum())}")
    del got

# ---------------------------------------------------------------------------------------------
# The three bias stacks, with real weights: is blocking them bit-exact?
print("\n--- bias stacks with real weights ---")
z_after = ref
for name, layers, x in (("token_trans_proj_z",
                         model.diffusion_conditioning.token_trans_proj_z, z_after),):
    h = layers[0][1].out_features
    ship = torch.cat([l(x) for l in layers], dim=-1)
    t_ship = tm(lambda: torch.cat([l(x) for l in layers], dim=-1))
    print(f"  {name}: {len(layers)} x {h},  shipped {1e3*t_ship:8.1f} ms")
    cw = x.shape[-1]
    flat = x.reshape(-1, cw)
    for rows in (1024, 2048, 4096, 8192):
        out = flat.new_empty(flat.shape[0], h * len(layers))
        def run(rows=rows, out=out):
            for s in range(0, flat.shape[0], rows):
                blk = flat[s:s + rows]
                for i, l in enumerate(layers):
                    out[s:s + rows, i * h:(i + 1) * h] = l(blk)
            return out.reshape(*x.shape[:-1], h * len(layers))
        got = run()
        t = tm(run)
        print(f"    rows={rows:6d}  {1e3*t:8.1f} ms  {t_ship/t:5.2f}x  "
              f"bit-exact {torch.equal(ship, got)}")
        del out, got
