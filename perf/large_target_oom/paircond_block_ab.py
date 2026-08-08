#!/usr/bin/env python3
"""A/B equivalence probe for the row-blocked _diffusion_pair_cond chain
(PAIRCOND_BLOCK_BYTES gate, protenix.py). Synthetic OpenDDE-scale diffusion-
conditioning weights, one z/relp draw, run the chain unblocked vs row-blocked
and report bit-exactness; an unblocked-vs-unblocked control measures the
run-to-run determinism floor.

    TT_VISIBLE_DEVICES=0 python3 perf/large_target_oom/paircond_block_ab.py
"""
import types

import torch
import ttnn

from tt_bio import protenix as P
from tt_bio import protenix_weights as PW
from tt_bio.tenstorrent import get_device

C = "diffusion_module.diffusion_conditioning."
# N=1888 (z = 2.75 GiB) is the production scale where the default gate fires, but the
# UNBLOCKED reference leg needs ~12.8 GiB there and OOMs a fresh 12 GiB Wormhole chip
# (measured: refused 912 MB in the transition concat at 11.87 GiB allocated). N=1600 keeps
# the production c_z=384 width, exercises the multi-block path (rb = 192 rows), and fits
# both legs (~6 GiB peak unblocked). The gate is monkey-patched per leg, not the default.
N = 1600
CZ, CP = 384, 128
F_RELP = 73       # 2*(32+1) + 2*(2+1) + 1, the _generate_relp default dim


def build_stub(dev, ckc):
    torch.manual_seed(0)
    w = {
        C + "relpe.linear_no_bias.weight": torch.randn(CP, F_RELP) * 0.05,
        C + "layernorm_z_trunk.weight": torch.randn(CZ) * 0.1 + 1.0,
        C + "linear_no_bias_z_trunk.weight": torch.randn(CP, CZ) * 0.05,
        C + "layernorm_z.weight": torch.randn(2 * CP) * 0.1 + 1.0,
        C + "linear_no_bias_z.weight": torch.randn(CP, 2 * CP) * 0.05,
    }
    for nm in ("transition_z1", "transition_z2"):
        w[C + nm + ".layernorm1.weight"] = torch.randn(CP) * 0.1 + 1.0
        w[C + nm + ".layernorm1.bias"] = torch.randn(CP) * 0.05
        w[C + nm + ".linear_no_bias_a.weight"] = torch.randn(4 * CP, CP) * 0.04
        w[C + nm + ".linear_no_bias_b.weight"] = torch.randn(4 * CP, CP) * 0.04
        w[C + nm + ".linear_no_bias.weight"] = torch.randn(CP, 4 * CP) * 0.04
    diff = types.SimpleNamespace(
        _up=lambda t: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                      dtype=ttnn.bfloat16),
        dtype=ttnn.bfloat16, _diffusion_fp32=False, dev=dev)
    return types.SimpleNamespace(_w=w, diffusion=diff, compute_kernel_config=ckc,
                                 _to_host=P.Protenix._to_host)


def run(stub, z0, relp0, gate):
    P.PAIRCOND_BLOCK_BYTES = gate
    z = ttnn.from_torch(z0, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=ttnn.bfloat16)
    out = P.Protenix._diffusion_pair_cond(stub, z, relp0)
    ttnn.deallocate(z)
    return out


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    stub = build_stub(dev, ckc)
    torch.manual_seed(1)
    z0 = torch.randn(1, N, N, CZ, dtype=torch.bfloat16)
    relp0 = torch.randint(0, 2, (N, N, F_RELP)).to(torch.bfloat16)
    print(f"N={N} z_bytes={N * N * CZ * 2 / 2 ** 30:.2f} GiB", flush=True)

    u0 = run(stub, z0, relp0, 10 ** 12)   # unblocked
    u1 = run(stub, z0, relp0, 10 ** 12)   # determinism control
    b0 = run(stub, z0, relp0, 0)          # row-blocked
    print("passes done", flush=True)

    def rep(a, b, tag):
        d = (a - b).abs().max().item()
        eq = torch.equal(a, b)
        pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        print(f"{tag}: bitexact={eq} maxabs={d:.6f} pcc={pcc:.8f}", flush=True)

    rep(u0, u1, "unblocked vs unblocked (determinism floor)")
    rep(u0, b0, "unblocked vs row-blocked")


if __name__ == "__main__":
    main()
