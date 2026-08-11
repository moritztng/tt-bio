"""Numerical error of whichever silu LLK the current TT_METAL_RUNTIME_ROOT supplies.

Runs the SAME fused-activation program the fold runs -- `:2528`'s [1,32,512,256] x [256,1024]
with core_grid=CORE_GRID_MAIN -- and makes the SFPU see the input verbatim by using a weight
built from four stacked identities, so out[..., :256] is silu(x) elementwise.

The previous version omitted core_grid. ttnn then emitted matmul + a standalone ttnn.silu, which
hardcodes math_approx_mode=false, so both LLK patches were no-ops and all three arms returned
byte-identical error figures (state doc S137). This script therefore refuses to report a number
until objdump confirms the activation actually fused into the bmm kernel.
"""
import argparse, glob, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

OBJDUMP = "/home/ttuser/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
M, K, N, B = 512, 256, 1024, 32


def sfpu_count(cache_root):
    """Count SFPU mnemonics in the fused-bias-activation compute kernel's trisc1 image."""
    hits = glob.glob(os.path.join(cache_root, "**",
                                  "bmm_large_block_zm_fused_bias_activation", "**", "trisc1.elf"),
                     recursive=True)
    out = {"elf_candidates": hits}
    if not hits:
        return out
    elf = sorted(hits)[0]
    d = subprocess.run([OBJDUMP, "-d", elf], capture_output=True, text=True).stdout
    out["elf"] = elf
    out["sfpu_instrs"] = sum(1 for ln in d.splitlines() if "\tsfp" in ln)
    out["total_instrs"] = sum(1 for ln in d.splitlines() if "\t" in ln and ":\t" in ln)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--expect-sfpu", type=int, default=None)
a = ap.parse_args()

cache = os.environ.get("TT_METAL_CACHE", "<unset>")
dev = get_device()
rec = {"tag": a.tag,
       "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT", "<unset>"),
       "kernel_cache": cache,
       "loadavg": open("/proc/loadavg").read().strip(),
       "shape": {"M": M, "K": K, "N": N, "B": B}}
try:
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    rec["ckc"] = {"math_approx_mode": bool(ckc.math_approx_mode),
                  "fp32_dest_acc_en": bool(ckc.fp32_dest_acc_en)}

    torch.manual_seed(0)
    xt = torch.empty(B * M, K)
    half = (B * M) // 2
    xt[:half] = torch.randn(half, K)                                    # realistic post-layernorm draw
    xt[half:] = torch.linspace(-12, 12, (B * M - half) * K).reshape(-1, K)   # tails
    xt = xt.to(torch.bfloat16)

    wt = torch.zeros(K, N, dtype=torch.bfloat16)
    for j in range(N // K):
        wt[:, j * K:(j + 1) * K] = torch.eye(K, dtype=torch.bfloat16)

    x = ttnn.from_torch(xt.reshape(1, B, M, K), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    o = ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc,
                    memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                    core_grid=CORE_GRID_MAIN)
    got = ttnn.to_torch(o)[0, :, :, :K].reshape(B * M, K).float()
    ttnn.deallocate(o); ttnn.deallocate(x); ttnn.deallocate(w)
    ttnn.synchronize_device(dev)

    ref = torch.nn.functional.silu(xt.float())
    err = (got - ref).abs()
    den = ref.abs().clamp(min=1e-3)
    core = xt.abs() <= 4
    g, r = got.flatten(), ref.flatten()
    rec.update({
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "rms_err": float((err ** 2).mean().sqrt()),
        "max_rel_err": float((err / den).max()),
        "pcc": float(torch.corrcoef(torch.stack([g, r]))[0, 1]),
        "ref_absmax": float(ref.abs().max()),
        "max_abs_err_core": float(err[core].max()),
        "rms_err_core": float((err[core] ** 2).mean().sqrt()),
        "n_core": int(core.sum()),
    })
except Exception as e:
    rec["error"] = repr(e)

rec["objdump"] = sfpu_count(cache) if cache != "<unset>" else {"skipped": "no TT_METAL_CACHE"}
n = rec["objdump"].get("sfpu_instrs")
rec["fusion_ok"] = bool(n)          # 0 SFPU instrs in the bmm kernel == the activation did not fuse
if a.expect_sfpu is not None:
    rec["sfpu_expected"] = a.expect_sfpu
    rec["sfpu_match"] = (n == a.expect_sfpu)
json.dump(rec, open(a.out, "w"), indent=2)
print(json.dumps({k: rec[k] for k in rec if k != "objdump"}, indent=2))
print("objdump:", {k: v for k, v in rec["objdump"].items() if k != "elf_candidates"})
