#!/usr/bin/env python3
"""Which fused triangle-attention kernels serve or decline at each width, and why.

Runs one call per (S, c_z x heads, start/ending, lever arm) and reads every served/declined counter the
triangle-attention kernels keep (triatt_ln, triatt_qkv, triatt_sdpa). Usage: declines.py --out d.jsonl
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import torch, ttnn  # noqa: E401
import tt_bio.tenstorrent as T
from tt_bio import triatt_ln as TL, triatt_qkv as TQ, triatt_sdpa as TS
from bench import ARMS, HD, weights

COUNTERS = {"ln": (TL.STATS, TL.REJECTS), "qkv": (TQ.STATS, TQ.REJECTS), "tail": (TQ.TAIL_STATS, TQ.TAIL_REJECTS),
            "qkvg": (TQ.QKVG_STATS, TQ.QKVG_REJECTS), "qkvgb": (TQ.QKVGB_STATS, TQ.QKVGB_REJECTS),
            "gop": (TQ.GOP_STATS, {}), "sdpa": (TS.STATS, TS.REJECTS), "sdpa_gate": (TS.GATE_STATS, TS.GATE_REJECTS),
            "sdpa_fuse": ([], TS.FUSE_REJECTS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--widths", default="256x8,384x12,320x10,448x14")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    for S in map(int, a.sizes.split(",")):
        for wh in a.widths.split(","):
            c, H = map(int, wh.split("x"))
            sd = weights(c, H, 7)
            x = torch.randn(1, S, S, c).bfloat16()
            for ending in (False, True):
                for arm, (lv, wide) in ARMS.items():
                    rec = {"S": S, "c_z": c, "heads": H, "ending": ending, "arm": arm}
                    try:
                        T._MM_DEFAULT_WIDE = wide
                        with T.levers(lv):
                            mod = T.TriangleAttention(HD, H, ending, sd, ckc, softmax_site="pairformer")
                            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            for st, rj in COUNTERS.values():
                                st[:] = [0] * len(st); rj.clear()
                            ttnn.deallocate(mod(xt)); ttnn.synchronize_device(dev); ttnn.deallocate(xt)
                        for k, (st, rj) in COUNTERS.items():
                            if any(st) or rj:
                                rec[k] = {"stats": list(st), "rejects": {str(kk): v for kk, v in rj.items()}}
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                    print(json.dumps(rec), flush=True)
                    with a.out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
    T.cleanup()


if __name__ == "__main__":
    main()
