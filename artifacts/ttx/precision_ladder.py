#!/usr/bin/env python3
"""Per-release churn on the precision contract, for the accuracy-slide bisect.

`ttx-version-recon` measured a GRADED degradation of the 298 aa control across releases --
plDDT 0.9132 at 0.68.0, 0.5764 at 0.73.1, 0.3597 at 0.78.0 -- and concluded it compounds rather
than breaking in one commit. Its next step is a version ladder. This prices each release step by
how much it moved the code that decides how a DeviceComputeKernelConfig is interpreted and how a
tile is packed, so the ladder can be walked in a useful order instead of bisected blind.

This is a POINTER, not a finding: churn is where to look, never why something broke.
"""
import subprocess
from pathlib import Path

MIRROR = Path(__file__).resolve().parent.parent.parent / ".ttm.git"

RELEASES = ["v0.68.0", "v0.69.0", "v0.70.1", "v0.71.2", "v0.72.0", "v0.73.1",
            "v0.74.0", "v0.75.0", "v0.76.0", "v0.77.0", "v0.78.0"]

# How a config becomes a kernel's rounding and accumulation behaviour.
CONTRACT = [
    "ttnn/cpp/ttnn/operations/core/compute_kernel/",   # DeviceComputeKernelConfig itself
    "tt_metal/impl/data_format/",                      # bfp8/bfp4 pack, tile, block float
    "tt_metal/jit_build/data_format.cpp",              # format resolution at build time
    "tt_metal/jit_build/data_format.hpp",
    "tt_metal/hw/inc/api/compute/reconfig_data_format.h",
    "tt_metal/hw/inc/api/compute/pack.h",
    "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp",       # DST width and sync
]
# The ops tt-bio passes explicit configs to, which is where version-recon's probe is going.
CONFIGURED_OPS = [
    "ttnn/cpp/ttnn/operations/normalization/layernorm/",
    "ttnn/cpp/ttnn/operations/normalization/rmsnorm/",
    "ttnn/cpp/ttnn/operations/normalization/softmax/",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/",
    "ttnn/cpp/ttnn/operations/matmul/",
]


def churn(a, b, paths):
    out = subprocess.run(["git", "-C", str(MIRROR), "diff", "--numstat", a, b, "--", *paths],
                         capture_output=True, text=True).stdout
    n = f = 0
    for ln in out.splitlines():
        x, y, _ = ln.split("\t", 2)
        if x == "-":
            continue
        n += int(x) + int(y)
        f += 1
    return n, f


if __name__ == "__main__":
    print(f"{'step':22} {'contract':>14} {'configured ops':>18}")
    for a, b in zip(RELEASES, RELEASES[1:]):
        cn, cf = churn(a, b, CONTRACT)
        on, of = churn(a, b, CONFIGURED_OPS)
        print(f"{a+' -> '+b:22} {str(cn)+'L/'+str(cf)+'f':>14} {str(on)+'L/'+str(of)+'f':>18}")
    cn, _ = churn(RELEASES[0], RELEASES[-1], CONTRACT)
    on, _ = churn(RELEASES[0], RELEASES[-1], CONFIGURED_OPS)
    print(f"\n{'0.68.0 -> 0.78.0 total':22} {str(cn)+'L':>14} {str(on)+'L':>18}")
