#!/usr/bin/env python3
"""Price the biggest unported host stage of the Boltz-2 fold against the device.

`b2x-host-residual` closed the 512 aa fold`s zero-device host block at 2.586 s and named
`DiffusionConditioning` as 0.920 s of it, 35.6 %: a torch module with no ttnn implementation at
all, 120 GFLOP of dense fp32 matmul running on 8 CPU cores while a Blackhole waits. Its two
largest parts are

  pairwise_conditioner   cat(z_trunk, relpos) -> LayerNorm -> LinearNoBias -> 2 x (Transition + res)
  token_trans_proj_z     24 x (LayerNorm(128) + Linear(128, 8, bias=False)), concatenated

and both are nothing but ttnn ops. This screen measures, at the fold`s real 512-token shapes and
with the real checkpoint weights, three things per part: the host seconds, the device seconds, and
the host<->device transfer seconds that a port would add. That is the whole lever, priced before a
line of engine code is written.

It also reports the numerics gap, because the port is NOT bit-exact: bf16 device math against fp32
torch. The gap here is a screen; the structural cost is the `cdk2x2_298` control in the fold.

No featurizer, no MSA, no fold: the stages timed are shape-bound, not value-bound, and the weights
come straight out of the checkpoint. z_trunk / relpos are synthetic at the model`s shapes, which is
what `hostpath_cost.py` does for the same stages.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_grad_enabled(False)

CKPT = os.environ.get("B2Z_CKPT", str(Path.home() / ".boltz" / "boltz2_conf.ckpt"))
PFX = "diffusion_conditioning."


def med(fn, reps, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), [round(x, 5) for x in ts]


def gap(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().flatten(), b.float().flatten()
    d = (a - b).abs()
    scale = a.abs().mean().clamp_min(1e-12)
    return {"max_abs": round(d.max().item(), 6),
            "mean_abs": round(d.mean().item(), 8),
            "rel_mean": round((d.mean() / scale).item(), 7),
            "ref_mean_abs": round(a.abs().mean().item(), 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"),
                    help="device math dtype. fp32 is the accuracy-safe arm")
    a = ap.parse_args()

    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "host": os.uname().nodename, "torch": torch.__version__,
                   "threads": torch.get_num_threads(), "cpus": os.cpu_count(),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "loadavg": open("/proc/loadavg").read().split()[:3],
                   "tokens": a.tokens, "reps": a.reps, "ckpt": CKPT}}

    # --- the real weights, without building the whole model ---------------------------------
    t0 = time.perf_counter()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sub = {k[len("model.") + len(PFX):]: v for k, v in sd.items()
           if k.startswith("model." + PFX)} or {
           k[len(PFX):]: v for k, v in sd.items() if k.startswith(PFX)}
    if not sub:
        print("no diffusion_conditioning weights found; keys look like:", flush=True)
        for k in list(sd)[:20]:
            print("   ", k, flush=True)
        return 2
    out["env"]["ckpt_load_s"] = round(time.perf_counter() - t0, 2)
    token_z = int(sub["pairwise_conditioner.transitions.0.norm.weight"].shape[0])
    n_trans = 1 + max(int(k.split(".")[2]) for k in sub
                      if k.startswith("pairwise_conditioner.transitions."))
    n_bias = 1 + max(int(k.split(".")[1]) for k in sub if k.startswith("token_trans_proj_z."))
    heads = int(sub["token_trans_proj_z.0.1.weight"].shape[0])
    hidden = int(sub["pairwise_conditioner.transitions.0.fc1.weight"].shape[0])
    out["shape"] = {"token_z": token_z, "transitions": n_trans, "bias_layers": n_bias,
                    "bias_heads": heads, "transition_hidden": hidden, "n": a.tokens}
    print(f"token_z {token_z}  transitions {n_trans}  bias layers {n_bias} x {heads} heads  "
          f"hidden {hidden}", flush=True)

    # --- host arm: the shipped modules, shipped code path -----------------------------------
    from tt_bio.boltz2 import PairwiseConditioning, _bias_stack
    import torch.nn as nn

    pc = PairwiseConditioning(token_z=token_z, dim_token_rel_pos_feats=token_z,
                              num_transitions=n_trans,
                              transition_expansion_factor=hidden // token_z).eval()
    pc.load_state_dict({k[len("pairwise_conditioner."):]: v for k, v in sub.items()
                        if k.startswith("pairwise_conditioner.")})
    bs = nn.ModuleList([nn.Sequential(nn.LayerNorm(token_z),
                                      nn.Linear(token_z, heads, bias=False))
                        for _ in range(n_bias)]).eval()
    bs.load_state_dict({k[len("token_trans_proj_z."):]: v for k, v in sub.items()
                        if k.startswith("token_trans_proj_z.")})

    n = a.tokens
    g = torch.Generator().manual_seed(0)
    z_trunk = torch.randn(1, n, n, token_z, generator=g)
    relpos = torch.randn(1, n, n, token_z, generator=g)

    t_pc, all_pc = med(lambda: pc(z_trunk, relpos), a.reps)
    z_ref = pc(z_trunk, relpos)
    t_bs, all_bs = med(lambda: _bias_stack(bs, z_ref), a.reps)
    b_ref = _bias_stack(bs, z_ref)
    out["host_s"] = {"pairwise_conditioner": round(t_pc, 5), "token_trans_bias": round(t_bs, 5),
                     "both": round(t_pc + t_bs, 5)}
    out["host_reps"] = {"pairwise_conditioner": all_pc, "token_trans_bias": all_bs}
    print(f"HOST  pairwise {1e3*t_pc:8.2f} ms   token_trans_bias {1e3*t_bs:8.2f} ms   "
          f"both {1e3*(t_pc+t_bs):8.2f} ms", flush=True)
    if a.host_only:
        a.out.write_text(json.dumps(out, indent=1))
        return 0

    # --- device arm --------------------------------------------------------------------------
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    out["env"]["grid"] = str(dev.compute_with_storage_grid_size())
    out["env"]["arch"] = T.arch_name()

    DT = ttnn.bfloat16 if a.dtype == "bf16" else ttnn.float32
    out["env"]["device_dtype"] = a.dtype

    def up(x, dtype=DT, tr=False):
        t = x.t() if tr else x
        return ttnn.from_torch(t.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # weights
    w = {k: up(v, tr=(v.dim() == 2)) for k, v in sub.items()
         if k.startswith(("pairwise_conditioner.", "token_trans_proj_z."))}

    def tt_pairwise(z_t, rp_t):
        x = ttnn.concat([z_t, rp_t], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xn = ttnn.layer_norm(x, weight=w["pairwise_conditioner.dim_pairwise_init_proj.0.weight"],
                             bias=w["pairwise_conditioner.dim_pairwise_init_proj.0.bias"],
                             epsilon=1e-5, compute_kernel_config=ckc,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        z = ttnn.linear(xn, w["pairwise_conditioner.dim_pairwise_init_proj.1.weight"],
                        compute_kernel_config=ckc, dtype=DT, core_grid=T.CORE_GRID_MAIN,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xn)
        for i in range(n_trans):
            p = f"pairwise_conditioner.transitions.{i}."
            zn = ttnn.layer_norm(z, weight=w[p + "norm.weight"], bias=w[p + "norm.bias"],
                                 epsilon=1e-5, compute_kernel_config=ckc,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            h1 = ttnn.linear(zn, w[p + "fc1.weight"], activation="silu",
                             compute_kernel_config=ckc, dtype=DT, core_grid=T.CORE_GRID_MAIN,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
            h2 = ttnn.linear(zn, w[p + "fc2.weight"], compute_kernel_config=ckc, dtype=DT,
                             core_grid=T.CORE_GRID_MAIN, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(zn)
            h1 = ttnn.multiply_(h1, h2)
            ttnn.deallocate(h2)
            d = ttnn.linear(h1, w[p + "fc3.weight"], compute_kernel_config=ckc, dtype=DT,
                            core_grid=T.CORE_GRID_MAIN, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(h1)
            z = ttnn.add_(z, d)
            ttnn.deallocate(d)
        return z

    def tt_bias(z_t):
        parts = []
        for i in range(n_bias):
            p = f"token_trans_proj_z.{i}."
            zn = ttnn.layer_norm(z_t, weight=w[p + "0.weight"], bias=w[p + "0.bias"],
                                 epsilon=1e-5, compute_kernel_config=ckc,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            parts.append(ttnn.linear(zn, w[p + "1.weight"], compute_kernel_config=ckc, dtype=DT,
                                     core_grid=T.CORE_GRID_MAIN,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(zn)
        b = ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p in parts:
            ttnn.deallocate(p)
        return b

    # --- the algebraic collapse of the bias stack ------------------------------------------
    # All 24 layers LayerNorm the SAME z, so the normalisation is computed 24 times and thrown
    # away 23 of them. With zhat = (z - mean) / std shared,
    #     layer_i(z) = (zhat * w_i + b_i) @ W_i.T
    #                = zhat @ (w_i * W_i).T + b_i @ W_i.T
    # so the whole stack is ONE affine-free LayerNorm plus ONE [C, heads*layers] Linear with a
    # constant bias. Exact in real arithmetic, not bit-exact in floating point (the products and
    # the summation order both move).
    W_eff = torch.cat([sub[f"token_trans_proj_z.{i}.1.weight"]
                       * sub[f"token_trans_proj_z.{i}.0.weight"] for i in range(n_bias)], dim=0)
    b_eff = torch.cat([sub[f"token_trans_proj_z.{i}.1.weight"]
                       @ sub[f"token_trans_proj_z.{i}.0.bias"] for i in range(n_bias)], dim=0)
    ln_fused = nn.LayerNorm(token_z, elementwise_affine=False).eval()

    def host_bias_fused(z):
        return torch.nn.functional.linear(ln_fused(z), W_eff, b_eff)

    t_bsf, all_bsf = med(lambda: host_bias_fused(z_ref), a.reps)
    b_fused_ref = host_bias_fused(z_ref)
    out["host_s"]["token_trans_bias_fused"] = round(t_bsf, 5)
    out["host_reps"]["token_trans_bias_fused"] = all_bsf
    out["numerics_host_fused"] = gap(b_ref, b_fused_ref)
    print(f"HOST  token_trans_bias FUSED {1e3*t_bsf:8.2f} ms  "
          f"(shipped {1e3*t_bs:8.2f} ms)  gap {out['numerics_host_fused']}", flush=True)

    w["fused.W"] = up(W_eff, tr=True)
    w["fused.b"] = ttnn.from_torch(b_eff.reshape(1, -1).contiguous(), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=DT)

    def tt_bias_fused(z_t):
        zn = ttnn.layer_norm(z_t, epsilon=1e-5, compute_kernel_config=ckc,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.linear(zn, w["fused.W"], bias=w["fused.b"], compute_kernel_config=ckc,
                        dtype=DT, core_grid=T.CORE_GRID_MAIN,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(zn)
        return b

    # transfers, timed on their own: this is what a port ADDS
    def t_up():
        x = up(z_trunk.reshape(n, n, token_z))
        ttnn.synchronize_device(dev)
        ttnn.deallocate(x)

    t_upload, all_upload = med(t_up, a.reps)

    z_t = up(z_trunk.reshape(n, n, token_z))
    rp_t = up(relpos.reshape(n, n, token_z))
    ttnn.synchronize_device(dev)

    def run_pc():
        z = tt_pairwise(z_t, rp_t)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(z)

    t_dev_pc, all_dev_pc = med(run_pc, a.reps, warm=2)
    z_dev = tt_pairwise(z_t, rp_t)
    ttnn.synchronize_device(dev)

    def run_bs():
        b = tt_bias(z_dev)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(b)

    t_dev_bs, all_dev_bs = med(run_bs, a.reps, warm=2)
    b_dev = tt_bias(z_dev)
    ttnn.synchronize_device(dev)

    def t_down():
        ttnn.to_torch(b_dev)

    t_download, all_download = med(t_down, a.reps)

    def run_bsf():
        b = tt_bias_fused(z_dev)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(b)

    t_dev_bsf, all_dev_bsf = med(run_bsf, a.reps, warm=2)
    b_dev_fused = tt_bias_fused(z_dev)
    ttnn.synchronize_device(dev)

    out["device_s"] = {"pairwise_conditioner": round(t_dev_pc, 5),
                       "token_trans_bias": round(t_dev_bs, 5),
                       "both": round(t_dev_pc + t_dev_bs, 5),
                       "upload_z": round(t_upload, 5),
                       "token_trans_bias_fused": round(t_dev_bsf, 5),
                       "download_bias": round(t_download, 5)}
    out["device_reps"] = {"pairwise_conditioner": all_dev_pc, "token_trans_bias": all_dev_bs,
                          "token_trans_bias_fused": all_dev_bsf,
                          "upload_z": all_upload, "download_bias": all_download}
    print(f"DEV   pairwise {1e3*t_dev_pc:8.2f} ms   token_trans_bias {1e3*t_dev_bs:8.2f} ms   "
          f"both {1e3*(t_dev_pc+t_dev_bs):8.2f} ms", flush=True)
    print(f"XFER  upload z {1e3*t_upload:8.2f} ms   download bias {1e3*t_download:8.2f} ms",
          flush=True)

    print(f"DEV   token_trans_bias FUSED {1e3*t_dev_bsf:8.2f} ms", flush=True)
    out["numerics"] = {"token_trans_bias_fused_dev": gap(
                           b_ref.reshape(n, n, heads * n_bias), ttnn.to_torch(b_dev_fused)),
                       "z": gap(z_ref.reshape(n, n, token_z), ttnn.to_torch(z_dev)),
                       "token_trans_bias": gap(b_ref.reshape(n, n, heads * n_bias),
                                               ttnn.to_torch(b_dev))}
    print("GAP   z", out["numerics"]["z"], flush=True)
    print("GAP   bias", out["numerics"]["token_trans_bias"], flush=True)

    host = out["host_s"]["both"]
    devtot = out["device_s"]["both"] + t_upload + t_download
    resident = t_dev_pc + t_dev_bsf          # z already on device, bias stays on device
    out["verdict"] = {
        "host_s": round(host, 4),
        "isolated_device_plus_xfer_s": round(devtot, 4),
        "isolated_saved_s": round(host - devtot, 4),
        "isolated_fold_ratio": round(23.841 / (23.841 - (host - devtot)), 4),
        "resident_device_s": round(resident, 4),
        "resident_saved_s": round(host - resident, 4),
        "resident_fold_ratio": round(23.841 / (23.841 - (host - resident)), 4),
        "note": "isolated adds an upload and a download that the INTEGRATED port does not pay: "
                "z_trunk is already on the device when the trunk ends (today's code downloads "
                "it) and bias_token is re-uploaded by the tt diffusion module (today's code "
                "downloads it and uploads it again). resident_* is the integrated price."}
    print("\nSAVED %8.2f ms isolated, %8.2f ms resident" % (
        1e3 * (host - devtot), 1e3 * (host - resident)), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
