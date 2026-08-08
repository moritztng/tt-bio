#!/usr/bin/env python3
"""Decisive probe: does host-mode trimul leave the device in a state that makes a
LATER tri_att compute differently? Run trimul_start+end in host mode, snapshot z,
then run tri_att_start on that z in device vs host mode and compare. Also the
reverse control (device trimul, then tri_att both ways).

    TT_VISIBLE_DEVICES=0 python3 perf/large_target_oom/triatt_after_trimul.py
"""
import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device
from perf.large_target_oom.pairlayer_capacity import build_layer


def trimul(dev, layer, z0, host):
    T.CONCAT_HOST_BYTES = 0 if host else 10 ** 12
    z = ttnn.from_torch(z0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zu = layer.triangle_multiplication_start(z, None)
    z = ttnn.add_(z, zu); ttnn.deallocate(zu)
    zu = layer.triangle_multiplication_end(z, None)
    z = ttnn.add_(z, zu); ttnn.deallocate(zu)
    return z   # device tensor, live


def att(dev, layer, z, host):
    T.CONCAT_HOST_BYTES = 0 if host else 10 ** 12
    out = layer.triangle_attention_start(z, None)
    got = ttnn.to_torch(out)
    ttnn.deallocate(out)
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
    print(f"N={N} c_z={c_z}", flush=True)

    # Host trimul, then tri_att device vs host on the SAME live z.
    z_live = trimul(dev, layer, z0, host=True)
    z_snap = ttnn.to_torch(z_live)
    a_dev = att(dev, layer, z_live, host=False)          # tri_att device on host-trimul z
    z_live2 = ttnn.from_torch(z_snap, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    a_host = att(dev, layer, z_live2, host=True)         # tri_att host on same values
    ttnn.deallocate(z_live); ttnn.deallocate(z_live2)
    d = (a_dev.float() - a_host.float()).abs().max().item()
    print(f"host-trimul -> tri_att device-vs-host: maxabs {d:.3e}", flush=True)

    # Control: device trimul, then tri_att device vs host.
    z_live = trimul(dev, layer, z0, host=False)
    z_snap = ttnn.to_torch(z_live)
    b_dev = att(dev, layer, z_live, host=False)
    z_live2 = ttnn.from_torch(z_snap, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b_host = att(dev, layer, z_live2, host=True)
    ttnn.deallocate(z_live); ttnn.deallocate(z_live2)
    d = (b_dev.float() - b_host.float()).abs().max().item()
    print(f"dev-trimul  -> tri_att device-vs-host: maxabs {d:.3e}", flush=True)

    # And: does host-trimul z itself match dev-trimul z? (sanity, bisect said yes)
    print(f"tri_att device after host-trimul vs after dev-trimul: maxabs {(a_dev.float()-b_dev.float()).abs().max().item():.3e}", flush=True)


if __name__ == "__main__":
    main()
