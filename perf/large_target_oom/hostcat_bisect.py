#!/usr/bin/env python3
"""Per-op bisect of the host-assembly path: drive one PairformerLayer's sub-ops
manually at N=1712, dumping z (and s) after each, for device-concat vs host-concat
mode in one process (the budget is read at call time, so it is switchable).

    TT_VISIBLE_DEVICES=0 python3 perf/large_target_oom/hostcat_bisect.py
"""
import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device
from perf.large_target_oom.pairlayer_capacity import build_layer


def run_pass(dev, layer, s0, z0, host):
    T._CONCAT_HOST_BYTES = 0 if host else 10 ** 12
    s = ttnn.from_torch(s0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(z0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    dumps = {}

    def snap(name, t):
        dumps[name] = ttnn.to_torch(t)

    zu = layer.triangle_multiplication_start(z, None)
    z = ttnn.add_(z, zu); ttnn.deallocate(zu); snap("trimul_start", z)
    zu = layer.triangle_multiplication_end(z, None)
    z = ttnn.add_(z, zu); ttnn.deallocate(zu); snap("trimul_end", z)
    zu = layer.triangle_attention_start(z, None)
    z = ttnn.add_(z, zu); ttnn.deallocate(zu); snap("tri_att_start", z)
    zu = layer.triangle_attention_end(z, None)
    z = ttnn.add_(z, zu); ttnn.deallocate(zu); snap("tri_att_end", z)
    zu = layer.transition_z(z)
    z = ttnn.add_(z, zu); ttnn.deallocate(zu); snap("transition_z", z)
    if layer.transform_s:
        s_norm = ttnn.layer_norm(s, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                                 epsilon=1e-5, compute_kernel_config=layer.compute_kernel_config)
        su = layer.attention_pair_bias(s_norm, z, seq_mask=None)
        ttnn.deallocate(s_norm)
        s = ttnn.add_(s, su); ttnn.deallocate(su); snap("apb", s)
        su = layer.transition_s(s)
        s = ttnn.add_(s, su); ttnn.deallocate(su); snap("transition_s", s)
    return dumps


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer("opendde", ckc)
    N = 1712
    torch.manual_seed(0)
    s0 = torch.randn(1, N, 384, dtype=torch.bfloat16)
    z0 = torch.randn(1, N, N, c_z, dtype=torch.bfloat16)
    print(f"layer built, N={N} c_z={c_z}", flush=True)

    ref = run_pass(dev, layer, s0, z0, host=False)
    print("device pass done", flush=True)
    got = run_pass(dev, layer, s0, z0, host=True)
    print("host pass done", flush=True)

    for name in ref:
        a, b = ref[name], got[name]
        same = torch.equal(a, b)
        d = (a.float() - b.float()).abs().max().item()
        print(f"{name:16s} {'BIT-EXACT' if same else f'DIFFERS maxabs {d:.3e} pcc {torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0,1]:.6f}'}", flush=True)


if __name__ == "__main__":
    main()
