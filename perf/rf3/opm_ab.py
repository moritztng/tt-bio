#!/usr/bin/env python3
"""The outer product mean without the outer product: is it right, and what does it buy.

Pass 4's MSA split found `OuterProductMean` at 27.06 ms per block at 512 aa, three times either
triangle multiplication on the same pair tensor, on an MSA of depth ONE. The op materialises the
whole [I, J, C, D] product (537 MB at 512 tokens) in order to project it down with `proj_o`, and
at small depth that is avoidable by reassociating:

    sum_cd a_ic b_jd W_cdk = sum_d b_jd (sum_c a_ic W_cdk)

2 I J D c_z FLOPs instead of 2 I J (C D) c_z -- 32x at C = D = 32 -- and no big intermediate.

Both arms here run on the SAME real device tensors out of a real trunk, and both are scored
against an fp64 evaluation of the same bf16 operands, so the input quantisation is common and
only the two reductions' own error is left. Arm A runs twice, so the A/A floor is measured.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def rel_rms(x, ref):
    return float((x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat_cache", default="/home/ttuser/rf3_perf_work/featcache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from perf.rf3.featcache import featurized
    fo = featurized(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                    n_recycles=max(args.n_recycles, 2), diffusion_batch_size=1,
                    seed=args.seed, cache_dir=args.feat_cache or None)
    f = fo["feats"]

    import ttnn
    from tt_bio import tenstorrent as tts
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from tt_bio.tenstorrent import get_device
    from perf.rf3.tt_rf3_bench import net_config

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=50, with_confidence=False)

    host = HostInputs.build(f, device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)

    msa_mod = tt.recycler.msa_module
    opm = msa_mod.outer_product
    # On-manifold input: the real subsampler on the real MSA row and the real single track.
    m = msa_mod.subsampler(host.msa_stack[0], s_inputs)
    m_keep = ttnn.to_torch(m).float()

    def call(small: bool):
        tts._OPM_SMALL_DEPTH = small
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = opm(ttnn.clone(m), None, None)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        t = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        return t, dt

    # ---- 1. the fp64 reference, on the same bf16 operands -------------------------------
    w = {k: v.double() for k, v in opm.weights.as_dict().items()}
    x = m_keep.double()
    x = x.reshape(x.shape[-3], x.shape[-2], x.shape[-1]) if x.dim() == 4 else x
    mc = F.layer_norm(x, (x.shape[-1],), w["norm.weight"], w["norm.bias"], eps=1e-5)
    a64 = mc @ w["proj_a.weight"].t() + w.get("proj_a.bias", 0.0)
    b64 = mc @ w["proj_b.weight"].t() + w.get("proj_b.bias", 0.0)
    S = a64.shape[0]
    zz = torch.einsum("sic,sjd->ijcd", a64, b64) / S
    ref = zz.reshape(zz.shape[0], zz.shape[1], -1) @ w["proj_o.weight"].t() + w["proj_o.bias"]

    # ---- 2. both arms, arm A twice ------------------------------------------------------
    A, ta = call(False)
    B, tb = call(True)
    A2, ta2 = call(False)
    sq = lambda t: t.reshape(t.shape[-3], t.shape[-2], t.shape[-1]).double()
    rep = {"aa": args.aa, "n_token": host.n_token, "msa_shape": tuple(m.shape),
           "out_shape": tuple(A.shape),
           "rel_rms_vs_fp64": {"materialised": rel_rms(sq(A), ref),
                               "reassociated": rel_rms(sq(B), ref)},
           "max_abs_diff_arms": float((sq(A) - sq(B)).abs().max()),
           "max_abs_diff_AA": float((sq(A) - sq(A2)).abs().max()),
           "ref_rms": float(ref.pow(2).mean().sqrt())}
    rep["accuracy_ratio_reassoc_over_mat"] = (
        rep["rel_rms_vs_fp64"]["reassociated"] / rep["rel_rms_vs_fp64"]["materialised"])

    # ---- 3. timing, medians of --reps warm calls ---------------------------------------
    times = {}
    for name, small in (("materialised", False), ("reassociated", True), ("materialised_again", False)):
        ts = []
        for _ in range(args.reps):
            _, dt = call(small)
            ts.append(dt)
        times[name] = round(statistics.median(ts) * 1e3, 4)
    rep["ms_per_call"] = times
    rep["speedup"] = round(times["materialised"] / times["reassociated"], 4)
    rep["aa_floor"] = round(times["materialised"] / times["materialised_again"], 4)

    # ---- 4. the whole MSA module, both arms -------------------------------------------
    def module(small: bool):
        tts._OPM_SMALL_DEPTH = small
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = msa_mod(host.msa_stack[0], ttnn.clone(z_init), s_inputs)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        t = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        return t, dt

    module(False)
    zA, mta = module(False)
    zB, mtb = module(True)
    zA2, mta2 = module(False)
    rep["module_s"] = {"materialised": round(mta, 4), "reassociated": round(mtb, 4),
                       "materialised_again": round(mta2, 4)}
    rep["module_speedup"] = round(mta / mtb, 4)
    rep["module_aa_floor"] = round(mta / mta2, 4)
    rep["module_z_rel_rms_arms"] = rel_rms(zB.double(), zA.double())
    rep["module_z_rel_rms_AA"] = rel_rms(zA2.double(), zA.double())
    tts._OPM_SMALL_DEPTH = False
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
