#!/usr/bin/env python3
"""Single TriangleAttention op A/B at large-N: one op, real weights, random z.

Compares this checkout's chunked path against a forced-unchunked reference by
running the SAME code with SEQ_LEN_MORE_CHUNKING raised above S (reference) and
at its normal value (chunked). Any difference is then attributable to the
chunking logic in this file alone.

    TT_VISIBLE_DEVICES=3 python3 triatt_probe.py --model opendde --n 1712
"""
import argparse

import torch

import ttnn


def run(dev, ckc, ending, z_host, weights, n):
    from tt_bio.tenstorrent import TriangleAttention
    ta = TriangleAttention(384 // 12, 12, ending, weights, ckc)
    z = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = ta(z, None)
    o = ttnn.to_torch(out)
    ttnn.deallocate(out)
    ttnn.deallocate(z)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1712)
    args = ap.parse_args()
    from tt_bio.tenstorrent import get_device
    from tt_bio.opendde import load_opendde_checkpoint, route_opendde_weights

    CKPT = ("/home/ttuser/.cache/huggingface/hub/models--aurekaresearch--OpenDDE/"
            "snapshots/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt")
    routed = route_opendde_weights(load_opendde_checkpoint(CKPT, abag=True))
    blk = {k[len("layers.0."):]: v for k, v in routed["refiner"].items()
           if k.startswith("layers.0.")}

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    torch.manual_seed(0)
    z_host = torch.randn(1, args.n, args.n, 384)

    import tt_bio.tenstorrent as T
    real_thresh = T.SEQ_LEN_MORE_CHUNKING
    for ending in (False, True):
        scope = "tri_att_end" if ending else "tri_att_start"
        w = T.WeightScope(blk).child(scope, "mha.")
        T.SEQ_LEN_MORE_CHUNKING = 10 ** 9   # force unchunked reference
        ref = run(dev, ckc, ending, z_host, w, args.n)
        T.SEQ_LEN_MORE_CHUNKING = real_thresh
        got = run(dev, ckc, ending, z_host, w, args.n)
        d = (ref.float() - got.float()).abs()
        pcc = torch.corrcoef(torch.stack([ref.float().flatten(), got.float().flatten()]))[0, 1].item()
        print(f"ending={ending}: maxabs {d.max().item():.4e} "
              f"rel_med {(d / ref.float().abs().clamp(min=1)).median().item():.3e} PCC {pcc:.8f}")


if __name__ == "__main__":
    main()
