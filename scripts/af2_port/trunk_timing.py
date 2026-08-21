"""What does the float32 residual path cost the trunk, and does it still fit at production size?

`AF2PairBlock.rne_residual` routes each residual add through float32, so the pair intermediate is
twice as wide for the length of one add. Pass 9 priced it at +0.42 s over four passes on gate
wall clock -- which includes a model load, a host reference arm and, when something misses, a
whole second float32 arm. This times the two device stacks and nothing else, warm, and it does it
at more than the fixture's 208 tokens because that is the only length the residual path has ever
run at.

The inputs are synthetic. Timing does not care what the numbers are, and the two arms see the
same tensors, so the comparison is exact even though the fold is not.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=. \\
        env/bin/python3 scripts/af2_port/trunk_timing.py --tokens 208,512,848
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: The trunk's own dimensions, from `tt_bio.af2_reference`.
NUM_MSA_ROWS = 2


def one(model, tokens: int, passes: int, rne: bool) -> dict:
    import ttnn

    model.set_rne_residual(rne)
    generator = torch.Generator().manual_seed(0)
    dtype = model.trunk_dtype
    from tt_bio.af2_reference import C_EXTRA, C_M, C_Z

    msa = torch.randn(NUM_MSA_ROWS, tokens, C_M, generator=generator).to(dtype)
    pair = torch.randn(tokens, tokens, C_Z, generator=generator).to(dtype)
    extra = torch.randn(1, tokens, C_EXTRA, generator=generator).to(dtype)
    extra_mask = torch.zeros(1, tokens, dtype=dtype)
    msa_mask = torch.ones(NUM_MSA_ROWS, tokens, dtype=dtype)
    pair_mask = torch.ones(tokens, tokens, dtype=dtype)

    times = []
    for index in range(passes):
        start = time.perf_counter()
        z = model.extra_msa_stack(extra, pair, extra_mask, pair_mask)
        _, z = model.evoformer_stack(msa, z, msa_mask, pair_mask)
        ttnn.synchronize_device(model._device)
        times.append(time.perf_counter() - start)
    warm = times[1:] or times
    return {"tokens": tokens, "rne_residual": rne, "passes": passes,
            "first_s": times[0], "warm_s": sum(warm) / len(warm),
            "warm_min_s": min(warm), "all_s": times}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="208")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--params", default=None)
    args = ap.parse_args()

    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    from tap_gate import DEFAULT_PARAMS

    model = load_af2_device_model(load_af2_state_dict(args.params or DEFAULT_PARAMS),
                                  template=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for tokens in (int(t) for t in args.tokens.split(",")):
            for rne in (False, True):
                try:
                    rows.append(one(model, tokens, args.passes, rne))
                except Exception as error:            # OOM is a result, not a crash
                    rows.append({"tokens": tokens, "rne_residual": rne,
                                 "error": f"{type(error).__name__}: {error}"[:400]})
                print(json.dumps(rows[-1]), file=sys.stderr, flush=True)

    for tokens in sorted({r["tokens"] for r in rows}):
        pair = {r["rne_residual"]: r for r in rows if r["tokens"] == tokens}
        if all("warm_s" in r for r in pair.values()):
            pair[True]["overhead_vs_ttnn_add"] = pair[True]["warm_s"] / pair[False]["warm_s"]
    print(json.dumps({"mode": "af2ig_trunk_timing", "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
