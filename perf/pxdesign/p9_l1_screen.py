"""Does the fp32-softmax score block go L1-resident at AF2-IG's ragged token counts?

`AF2PairBlock` builds both `TriangleAttention`s with `fp32_softmax=True`, so the trunk's triangle
attention runs `_fp32_softmax_attention` -- the materialised O(S^3) score tensor. Main's S1/S2 work
makes that tensor's fp32 copy L1-resident, and it is dark for this model: the plan refuses on
`width % 32` read off the LOGICAL token count while the tensor sits in DRAM at its tile-PADDED
one, and every PXDesign token count is 16 mod 32. `TT_BIO_FP32_SOFTMAX_L1_PADDED` derives the plan
and the shard from the padded extent instead.

The flag is a module global, so the arms are flipped by rebinding it: one model, one weight load,
one allocator, arms interleaved, nothing but the score copy's memory config differs. Both the
counters and a digest of the trunk output come back per arm, because this lever is claimed to be
BIT-EXACT (the block partitions the leading dim, the softmax reduces over the last, and
`in0_block_w` is a function of tile counts and not of the batch a block height moves) and a
claim like that is worth a digest rather than an argument.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p9 \\
        PYTHONPATH=. python3 perf/pxdesign/p9_l1_screen.py --tokens 848 --passes 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "af2_port"))

NUM_MSA_ROWS = 2


def digest(t) -> str:
    return hashlib.sha256(t.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def one(model, tokens: int, passes: int, padded: bool) -> dict:
    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio.af2_reference import C_EXTRA, C_M, C_Z

    TT._FP32_SOFTMAX_L1_PADDED = padded
    # Every memo the plan latches into is keyed on a shape class the arms do not share, but clear
    # them anyway so an arm cannot inherit the other's refusal history.
    for key in TT.FP32_SOFTMAX_STATS:
        TT.FP32_SOFTMAX_STATS[key] = 0
    TT._FP32_SOFTMAX_L1_ROW_CAP.clear()
    TT._FP32_SOFTMAX_L1_FREE_ROW_CAP.clear()
    TT._FP32_SOFTMAX_L1_REFUSALS.clear()
    TT._fp32_softmax_l1_plan.cache_clear()

    generator = torch.Generator().manual_seed(0)
    dtype = model.trunk_dtype
    msa = torch.randn(NUM_MSA_ROWS, tokens, C_M, generator=generator).to(dtype)
    pair = torch.randn(tokens, tokens, C_Z, generator=generator).to(dtype)
    extra = torch.randn(1, tokens, C_EXTRA, generator=generator).to(dtype)
    extra_mask = torch.zeros(1, tokens, dtype=dtype)
    msa_mask = torch.ones(NUM_MSA_ROWS, tokens, dtype=dtype)
    pair_mask = torch.ones(tokens, tokens, dtype=dtype)

    times, z_digest, m_digest = [], None, None
    for index in range(passes):
        start = time.perf_counter()
        z = model.extra_msa_stack(extra, pair, extra_mask, pair_mask)
        m, z = model.evoformer_stack(msa, z, msa_mask, pair_mask)
        ttnn.synchronize_device(model._device)
        times.append(time.perf_counter() - start)
        if index == passes - 1:
            z_digest, m_digest = digest(ttnn.to_torch(z)), digest(ttnn.to_torch(m))
    warm = times[1:] or times
    return {"tokens": tokens, "l1_padded": padded, "passes": passes,
            "first_s": times[0], "warm_s": sum(warm) / len(warm),
            "warm_min_s": min(warm), "all_s": times,
            "z_sha16": z_digest, "m_sha16": m_digest,
            "fp32_softmax_stats": dict(TT.FP32_SOFTMAX_STATS),
            "l1_row_caps": {str(k): v for k, v in TT._FP32_SOFTMAX_L1_ROW_CAP.items()},
            "l1_free_row_caps": {str(k): v for k, v in TT._FP32_SOFTMAX_L1_FREE_ROW_CAP.items()}}


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
                rows.append({"tokens": args.tokens, "l1_padded": arm == "1",
                             "error": f"{type(error).__name__}: {error}"[:600]})
            print(json.dumps(rows[-1]), file=sys.stderr, flush=True)

    out = {"mode": "af2ig_fp32_softmax_l1_padded_screen", "rows": rows}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
