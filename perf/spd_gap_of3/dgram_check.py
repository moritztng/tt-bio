"""OpenFold3's trunk distogram on the card (OF3ConfidenceHead.distogram_device) against float64.

    python perf/spd_gap_of3/dgram_check.py --out out/dgram [--n 512] [--scale 30] [--ckpt of3-p2-155k.pt]

z ~ N(0, scale) at [1, N, N, 128] uploaded bf16 (the trunk pair's dtype), the real checkpoint's distogram weight.
Also block 0's attention-pair bias (_pair_bias_device) against float64. Reports max / rms abs error of the device logits and of the old host fp32 path against float64 on the same
bf16-rounded z, plus the bin argmax agreement, and the device call's wall time (median of --reps, synced).
"""
import argparse, json, os, statistics, time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ckpt", default=os.path.expanduser("~/spd/spd-shipflow/weights/of3-p2-155k.pt"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    from tt_bio.openfold3_fold import _sub
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    head = OF3ConfidenceHead(_sub(sd, "aux_heads"), dev, ckc)
    w = head._w["distogram.linear.weight"].double()
    del sd

    g = torch.Generator().manual_seed(0)
    z = (torch.randn(a.n, a.n, 128, generator=g) * a.scale).bfloat16()
    z_d = ttnn.from_torch(z.float()[None], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ref = torch.nn.functional.linear(z.double(), w); ref = ref + ref.transpose(0, 1)
    host = torch.nn.functional.linear(z.float(), w.float()); host = host + host.transpose(0, 1)

    times = []
    for _ in range(a.reps + 1):
        ttnn.synchronize_device(dev); t = time.perf_counter()
        out = head.distogram_device(z_d)
        times.append(time.perf_counter() - t)
    assert torch.isfinite(out).all(), "non-finite device distogram"
    err = lambda x: {"max_abs": float((x.double() - ref).abs().max()),
                     "rms": float(((x.double() - ref) ** 2).mean().sqrt()),
                     "argmax_agree": float((x.argmax(-1) == ref.argmax(-1)).double().mean())}
    rep = {"arch": str(dev.arch()), "n": a.n, "scale": a.scale, "ref_rms": float(ref.pow(2).mean().sqrt()),
           "device": err(out), "host_fp32": err(host), "device_ms_median": 1e3 * statistics.median(times[1:])}
    # Block 0's attention-pair bias (LN_z + linear_z, fp32 on device): same z, float64 reference.
    from tt_bio.openfold3_confidence import _BLK
    pfx = (_BLK % 0) + "attn_pair_bias."
    W = lambda k: head._w[pfx + k].double()
    zn = torch.nn.functional.layer_norm(z.double(), (z.shape[-1],), W("layer_norm_z.weight"),
                                        W("layer_norm_z.bias"), eps=1e-5)
    bref = torch.nn.functional.linear(zn, W("linear_z.weight")).permute(2, 0, 1)
    bdev = head._pair_bias_device(z_d, 0, a.n).double()
    rep["pair_bias"] = {"ref_rms": float(bref.pow(2).mean().sqrt()),
                        "max_abs": float((bdev - bref).abs().max()),
                        "rms": float(((bdev - bref) ** 2).mean().sqrt())}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dgram.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
