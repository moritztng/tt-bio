"""Does the fused HiFi triangle attention serve AF2-IGs trunk, and what does it cost?

`AF2PairBlock` builds both `TriangleAttention`s with `fp32_softmax=True`, which routes them to
`_fp32_softmax_attention` -- the materialised O(S^3) score tensor. `TT_BIO_TRIATT_FUSED_HIFI`
sends exactly those blocks to the persistent-mask fused SDPA at `HiFi4 / fp32_dest_acc` instead.
The flag is a module global read once at import, so the arms are flipped by rebinding it: one
model, one weight load, one allocator, interleaved arms, so nothing but the attention path differs.

The screen reports whether the fused path is SERVED at all (`TRIATT_FUSED_HIFI_STATS` and
`triatt_sdpa.REJECTS`), because a decline is silent -- the caller falls back to the materialised
softmax and the arm reads as "flat".

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p8 \\
        PYTHONPATH=. python3 perf/pxdesign/p8_triatt_screen.py --tokens 848 --passes 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "af2_port"))

NUM_MSA_ROWS = 2


def one(model, tokens: int, passes: int, fused: bool) -> dict:
    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_sdpa as TS
    from tt_bio.af2_reference import C_EXTRA, C_M, C_Z

    TT._TRIATT_FUSED_HIFI = fused
    for key in TT.TRIATT_FUSED_HIFI_STATS:
        TT.TRIATT_FUSED_HIFI_STATS[key] = 0
    TS.REJECTS.clear()
    served0 = list(TS.STATS)

    generator = torch.Generator().manual_seed(0)
    dtype = model.trunk_dtype
    msa = torch.randn(NUM_MSA_ROWS, tokens, C_M, generator=generator).to(dtype)
    pair = torch.randn(tokens, tokens, C_Z, generator=generator).to(dtype)
    extra = torch.randn(1, tokens, C_EXTRA, generator=generator).to(dtype)
    extra_mask = torch.zeros(1, tokens, dtype=dtype)
    msa_mask = torch.ones(NUM_MSA_ROWS, tokens, dtype=dtype)
    pair_mask = torch.ones(tokens, tokens, dtype=dtype)

    times = []
    for _ in range(passes):
        start = time.perf_counter()
        z = model.extra_msa_stack(extra, pair, extra_mask, pair_mask)
        _, z = model.evoformer_stack(msa, z, msa_mask, pair_mask)
        ttnn.synchronize_device(model._device)
        times.append(time.perf_counter() - start)
    warm = times[1:] or times
    return {"tokens": tokens, "fused_hifi": fused, "passes": passes,
            "first_s": times[0], "warm_s": sum(warm) / len(warm),
            "warm_min_s": min(warm), "all_s": times,
            "hifi_stats": dict(TT.TRIATT_FUSED_HIFI_STATS),
            "persistent_mask_delta": [TS.STATS[0] - served0[0], TS.STATS[1] - served0[1]],
            "rejects": {str(k): v for k, v in TS.REJECTS.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=848)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--arms", default="0,1,0,1")
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    from tap_gate import DEFAULT_PARAMS

    model = load_af2_device_model(load_af2_state_dict(args.params or DEFAULT_PARAMS),
                                  template=False)
    model.eval()
    model.set_rne_residual(True)

    rows = []
    with torch.no_grad():
        for arm in args.arms.split(","):
            try:
                rows.append(one(model, args.tokens, args.passes, arm == "1"))
            except Exception as error:            # an OOM or an L1 refusal is a result
                rows.append({"tokens": args.tokens, "fused_hifi": arm == "1",
                             "error": f"{type(error).__name__}: {error}"[:600]})
            print(json.dumps(rows[-1]), file=sys.stderr, flush=True)

    out = {"mode": "af2ig_triatt_fused_hifi_screen", "rows": rows}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
