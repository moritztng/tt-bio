#!/usr/bin/env python3
"""Diff-pattern probe for the tri_att host-assembly bug: run triangle_attention_start
in device vs host mode on the same z, dump the update, and report WHERE the values
diverge (which rows) plus a device-vs-device determinism control.

    TT_VISIBLE_DEVICES=0 python3 perf/large_target_oom/triatt_hostcat_diff.py
"""
import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device
from perf.large_target_oom.pairlayer_capacity import build_layer


def att_out(dev, layer, z0, host):
    T.CONCAT_HOST_BYTES = 0 if host else 10 ** 12
    z = ttnn.from_torch(z0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = layer.triangle_attention_start(z, None)
    got = ttnn.to_torch(out)
    ttnn.deallocate(out)
    ttnn.deallocate(z)
    return got


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer("opendde", ckc)
    N = 1712
    torch.manual_seed(0)
    z0 = torch.randn(1, N, N, c_z, dtype=torch.bfloat16)
    print(f"N={N} c_z={c_z} chunk={T.TRIANGLE_ATT_CHUNK_SIZE}", flush=True)

    d0 = att_out(dev, layer, z0, host=False)
    d1 = att_out(dev, layer, z0, host=False)   # determinism control
    h0 = att_out(dev, layer, z0, host=True)
    print("passes done", flush=True)

    dd = (d0.float() - d1.float()).abs()
    dh = (d0.float() - h0.float()).abs()
    print(f"device-vs-device: maxabs {dd.max().item():.3e} (nondeterminism floor)", flush=True)
    print(f"device-vs-host:   maxabs {dh.max().item():.3e}", flush=True)
    # Per-row max diff over the row axis (dim 1) to see the boundary pattern.
    rowmax = dh[0].amax(dim=(1, 2))   # [N]
    bad = (rowmax > 1e-1).nonzero().flatten()
    print(f"rows with |diff|>0.1: {bad.numel()} of {N}", flush=True)
    if bad.numel():
        print(f"first bad rows: {bad[:20].tolist()}", flush=True)
        print(f"last bad rows:  {bad[-20:].tolist()}", flush=True)
    colmax = dh[0].amax(dim=(0, 2))   # [N]
    badc = (colmax > 1e-1).nonzero().flatten()
    print(f"cols with |diff|>0.1: {badc.numel()} of {N}", flush=True)
    if badc.numel():
        print(f"first bad cols: {badc[:20].tolist()}", flush=True)


if __name__ == "__main__":
    main()
