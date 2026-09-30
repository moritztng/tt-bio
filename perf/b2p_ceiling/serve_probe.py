#!/usr/bin/env python3
"""Which padded token axes does the fused-HiFi triangle attention serve at BindCraft 2's shape?

q/k/v [N, 4, N, 32] bf16 against a [1, 4, N, N] bias, i.e. triangle attention's own geometry with
batch = sequence, called through `_tri_att_sdpa_hifi` exactly as the Evoformer calls it (untaped:
under the tape the arm runs the same function for the forward value). Prints served/declined, the
pick, and the kernel's rejection reasons per axis.

Usage: serve_probe.py [--sizes 512,544,...] [--out serve_probe.json]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(n) for n in range(192, 865, 32)))
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("serve_probe.json"))
    a = ap.parse_args()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as TS
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
    dev = T.get_device()
    res = {"grid": list(T.COMPUTE_GRID_MAIN), "sizes": {}}
    for S in [int(x) for x in a.sizes.split(",")]:
        torch.manual_seed(0)
        mk = lambda *sh: ttnn.from_torch(torch.randn(*sh).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=dev,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v, b = mk(S, 4, S, 32), mk(S, 4, S, 32), mk(S, 4, S, 32), mk(1, 4, S, S)
        before = dict(T.TRIATT_FUSED_HIFI_STATS)
        TS.REJECTS.clear()
        o = T._tri_att_sdpa_hifi(q, k, v, b, 32 ** -0.5)
        rec = {"served": o is not None, "pick": T.TRIATT_FUSED_HIFI_PICKS.get((S, S)), "padded_to": getattr(T, "TRIATT_FUSED_HIFI_PADDED", {}).get(S),
               "stats_delta": {kk: T.TRIATT_FUSED_HIFI_STATS[kk] - before[kk] for kk in before},
               "rejects": {f"{r}": n for (r, _s), n in TS.REJECTS.items()}}
        res["sizes"][S] = rec
        print(S, rec, flush=True)
        for t in (q, k, v, b) + ((o,) if o is not None else ()):
            ttnn.deallocate(t)
    a.out.write_text(json.dumps(res, indent=1))
    T.cleanup()


if __name__ == "__main__":
    main()
