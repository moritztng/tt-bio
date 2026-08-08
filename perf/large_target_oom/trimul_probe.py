#!/usr/bin/env python3
"""Single TriangleMultiplication A/B: chunked tail (normal threshold) vs full-size
tail (threshold raised), both in THIS checkout. Bit-level attribution for F2b.

    TT_VISIBLE_DEVICES=3 python3 trimul_probe.py --n 1712
"""
import argparse

import torch

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1712)
    args = ap.parse_args()
    import tt_bio.tenstorrent as T
    from tt_bio.opendde import load_opendde_checkpoint, route_opendde_weights

    CKPT = ("/home/ttuser/.cache/huggingface/hub/models--aurekaresearch--OpenDDE/"
            "snapshots/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt")
    routed = route_opendde_weights(load_opendde_checkpoint(CKPT, abag=True))
    blk = {k[len("layers.0."):]: v for k, v in routed["refiner"].items()
           if k.startswith("layers.0.")}

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    z_host = torch.randn(1, args.n, args.n, 384)
    real_thresh = T.SEQ_LEN_MORE_CHUNKING

    for ending in (False, True):
        scope = "tri_mul_in" if ending else "tri_mul_out"
        tm = T.TriangleMultiplication(ending, T.WeightScope(blk).child(scope), ckc)

        def run():
            z = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            out = tm(z, None)
            o = ttnn.to_torch(out)
            ttnn.deallocate(out)
            ttnn.deallocate(z)
            return o

        T.SEQ_LEN_MORE_CHUNKING = 10 ** 9     # full-size tail reference
        ref = run()
        T.SEQ_LEN_MORE_CHUNKING = real_thresh  # row-blocked tail
        got = run()
        d = (ref.float() - got.float()).abs()
        pcc = torch.corrcoef(torch.stack([ref.float().flatten(), got.float().flatten()]))[0, 1].item()
        print(f"ending={ending}: {'BIT-EXACT' if torch.equal(ref, got) else 'DIFFERS'} "
              f"maxabs {d.max().item():.4e} rel_med "
              f"{(d / ref.float().abs().clamp(min=1)).median().item():.3e} PCC {pcc:.8f}", flush=True)


if __name__ == "__main__":
    main()
