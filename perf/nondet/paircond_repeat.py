#!/usr/bin/env python3
"""In-process repeat probe for the Protenix-v2 diffusion pair-conditioning chain.

The full-fold instruments (TT_PROTENIX_DUMP / TT_PROTENIX_PZPROBE) need a ~20 min fold per
sample and confound compute, allocation order and process state. This runs the SAME chain
K times inside ONE process on ONE fixed device input and reports, per op, whether the
output is bit-identical across repeats. That names the nondeterministic op directly.

Two axes are separated at every stage:
  compute repeat  -- run the op K times, compare output k vs output 0
  readback repeat -- read the SAME device tensor back twice, compare

compute differs + readback stable  -> the kernel is racy / order-unstable
readback differs                   -> the DRAM/PCIe readback of those addresses is unstable

Protenix-v2 pair-cond shapes (from the shipped checkpoint): c_z = c_z_pair_diffusion = 256
(no compression branch), relpe 139 -> 256, concat 512, linear_z 512 -> 256, then two
Transitions 256 -> 512 -> 256.

    TT_VISIBLE_DEVICES=<card> python3 perf/nondet/paircond_repeat.py [N] [K]
"""
import json
import os
import pathlib
import sys
import types

import torch
import ttnn

from tt_bio import protenix as P
from tt_bio import protenix_weights as PW
from tt_bio import tenstorrent as TT
from tt_bio.tenstorrent import Transition, get_device

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = sys.argv[3] if len(sys.argv) > 3 else None

C = "diffusion_module.diffusion_conditioning."
CZ = CP = 256
F_RELP = 139


def build_weights():
    torch.manual_seed(0)
    w = {
        C + "relpe.linear_no_bias.weight": torch.randn(CP, F_RELP) * 0.05,
        C + "layernorm_z.weight": torch.randn(2 * CP) * 0.1 + 1.0,
        C + "linear_no_bias_z.weight": torch.randn(CP, 2 * CP) * 0.05,
    }
    for nm in ("transition_z1", "transition_z2"):
        w[C + nm + ".layernorm1.weight"] = torch.randn(CP) * 0.1 + 1.0
        w[C + nm + ".layernorm1.bias"] = torch.randn(CP) * 0.05
        w[C + nm + ".linear_no_bias_a.weight"] = torch.randn(2 * CP, CP) * 0.04
        w[C + nm + ".linear_no_bias_b.weight"] = torch.randn(2 * CP, CP) * 0.04
        w[C + nm + ".linear_no_bias.weight"] = torch.randn(CP, 2 * CP) * 0.04
    return w


def diff_report(ref, got):
    """Bit-exactness + the spatial fingerprint used on the pc dump triple."""
    if torch.equal(ref, got):
        return dict(bitexact=True, maxabs=0.0, cells=0, rows=0)
    a, b = ref.float(), got.float()
    d = (a - b).abs()
    flat = d.reshape(-1, d.shape[-1])
    bad = (flat > 1e-3).any(-1)
    idx = bad.nonzero().flatten()
    rows = torch.unique(idx // (ref.shape[-2] if ref.dim() >= 3 else 1)).numel()
    return dict(bitexact=False, maxabs=float(d.max()), cells=int(bad.sum()), rows=int(rows))


def repeat_stage(name, fn, k, results):
    """Run fn() k times; compare every output to the first. Also double-read output 0."""
    outs = []
    for i in range(k):
        t = fn()
        ttnn.synchronize_device(get_device())
        if i == 0:
            r1 = ttnn.to_torch(t)
            r2 = ttnn.to_torch(t)
            readback = diff_report(r1, r2)
            keep = t
            outs.append(r1)
        else:
            outs.append(ttnn.to_torch(t))
            ttnn.deallocate(t)
    cmp = [diff_report(outs[0], o) for o in outs[1:]]
    ndiff = sum(0 if c["bitexact"] else 1 for c in cmp)
    worst = max(cmp, key=lambda c: c["maxabs"]) if cmp else dict(maxabs=0.0, cells=0, rows=0)
    rec = dict(stage=name, repeats=k, compute_differing=ndiff,
               compute_worst_maxabs=round(worst["maxabs"], 6),
               compute_worst_cells=worst["cells"], compute_worst_rows=worst["rows"],
               readback_bitexact=readback["bitexact"],
               readback_maxabs=round(readback["maxabs"], 6),
               out_absmax=round(float(outs[0].float().abs().max()), 4))
    results.append(rec)
    print(f"  {name:22s} compute {k - 1 - ndiff}/{k - 1} identical"
          f"  worst_maxabs={rec['compute_worst_maxabs']:<10} cells={worst['cells']:<5}"
          f"  readback_bitexact={readback['bitexact']}", flush=True)
    return keep


def main():
    dev = get_device()
    grid = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print(f"host={os.uname().nodename} card={os.environ.get('TT_VISIBLE_DEVICES')} "
          f"arch={dev.arch()} storage_grid={grid.x}x{grid.y} "
          f"CORE_GRID_MAIN={TT.COMPUTE_GRID_MAIN} N={N} K={K}", flush=True)

    w = build_weights()
    up = lambda t: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                   dtype=ttnn.bfloat16)

    torch.manual_seed(1)
    z0 = torch.randn(N, N, CZ, dtype=torch.bfloat16)
    relp0 = torch.randint(0, 2, (N, N, F_RELP)).to(torch.bfloat16)

    z_tt = up(z0)
    relp_tt = up(relp0)
    Wr = up(w[C + "relpe.linear_no_bias.weight"].t().contiguous())
    Wz = up(w[C + "linear_no_bias_z.weight"].t().contiguous())
    lnz_w = up(w[C + "layernorm_z.weight"])

    # Upload stability control: same host bytes uploaded twice must read back identically.
    up_ctl = diff_report(ttnn.to_torch(z_tt), ttnn.to_torch(up(z0)))
    print(f"  {'upload_control':22s} bitexact={up_ctl['bitexact']} "
          f"maxabs={up_ctl['maxabs']}", flush=True)

    results = []
    print("stage repeats (same device input, K runs, compare vs run 0):", flush=True)

    relpe = repeat_stage("relpe_linear", lambda: ttnn.linear(
        relp_tt, Wr, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
        core_grid=TT.CORE_GRID_MAIN), K, results)

    zc = repeat_stage("concat", lambda: ttnn.concat([z_tt, relpe], dim=-1), K, results)

    zn = repeat_stage("layer_norm", lambda: ttnn.layer_norm(
        zc, weight=lnz_w, epsilon=1e-5, compute_kernel_config=ckc), K, results)

    pb = repeat_stage("linear_z", lambda: ttnn.linear(
        zn, Wz, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
        core_grid=TT.CORE_GRID_MAIN), K, results)

    pz4 = ttnn.reshape(pb, (1, N, N, CP))
    trs = []
    for nm in ("transition_z1", "transition_z2"):
        sub = {k[len(C + nm + "."):]: v for k, v in w.items() if k.startswith(C + nm + ".")}
        trs.append(Transition(PW.remap_transition(sub), ckc, dtype=ttnn.bfloat16))

    t1 = repeat_stage("transition_z1", lambda: trs[0](pz4), K, results)
    s1 = repeat_stage("add_z1", lambda: ttnn.add(pz4, t1), K, results)
    t2 = repeat_stage("transition_z2", lambda: trs[1](s1), K, results)
    repeat_stage("add_z2", lambda: ttnn.add(s1, t2), K, results)

    # Whole chain end to end, fresh uploads each pass -- what a fold actually does.
    def chain():
        z = up(z0)
        r = ttnn.linear(up(relp0), Wr, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                        core_grid=TT.CORE_GRID_MAIN)
        c = ttnn.concat([ttnn.reshape(z, (N, N, CZ)), r], dim=-1)
        c = ttnn.layer_norm(c, weight=lnz_w, epsilon=1e-5, compute_kernel_config=ckc)
        p = ttnn.linear(c, Wz, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                        core_grid=TT.CORE_GRID_MAIN)
        p = ttnn.reshape(p, (1, N, N, CP))
        for t in trs:
            p = ttnn.add(p, t(p))
        return p

    print("full chain (fresh uploads per pass):", flush=True)
    repeat_stage("full_paircond", chain, K, results)

    rec = dict(host=os.uname().nodename, card=os.environ.get("TT_VISIBLE_DEVICES"),
               storage_grid=f"{grid.x}x{grid.y}", core_grid_main=list(TT.COMPUTE_GRID_MAIN),
               N=N, K=K, upload_control_bitexact=up_ctl["bitexact"], stages=results)
    if OUT:
        p = pathlib.Path(OUT)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=2))
        print(f"wrote {OUT}", flush=True)
    unstable = [r["stage"] for r in results if r["compute_differing"]]
    print(f"VERDICT unstable_stages={unstable or 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
