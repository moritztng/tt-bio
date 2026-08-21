"""Where the float32 residual temporaries live, what it costs, and that it changes nothing.

`trunk_timing.py` at 208/512/848 found the float32 residual path OOMs at 512 tokens and runs at
848. That is a window, not a cap: the bfloat16 pair is 64 MB at 512 and fits L1, so its float32
copy inherits L1 too and needs 128 MB across 130 banks with 943 KB free per bank; at 848 the
bfloat16 pair is 184 MB, is already in DRAM, and the float32 copy follows it there.

`AF2PairBlock.rne_wide_dram` puts the temporaries in DRAM at every length and returns the result
to the input's own memory config. Placement is not arithmetic, so the two arms have to agree bit
for bit wherever both run, and this checks that rather than asserting it.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=. \\
        env/bin/python3 scripts/af2_port/rne_placement.py --tokens 208,512,848
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

NUM_MSA_ROWS = 2

#: (label, rne_residual, rne_wide_dram). The bfloat16 arm is the cost denominator.
ARMS = (("bf16_add", False, False), ("wide_l1", True, False), ("wide_dram", True, True))


def one(model, tokens: int, passes: int, rne: bool, dram: bool) -> dict:
    import ttnn

    model.set_rne_residual(rne)
    model.set_rne_wide_dram(dram)
    generator = torch.Generator().manual_seed(0)
    dtype = model.trunk_dtype
    from tt_bio.af2_reference import C_EXTRA, C_M, C_Z

    msa = torch.randn(NUM_MSA_ROWS, tokens, C_M, generator=generator).to(dtype)
    pair = torch.randn(tokens, tokens, C_Z, generator=generator).to(dtype)
    extra = torch.randn(1, tokens, C_EXTRA, generator=generator).to(dtype)
    extra_mask = torch.zeros(1, tokens, dtype=dtype)
    msa_mask = torch.ones(NUM_MSA_ROWS, tokens, dtype=dtype)
    pair_mask = torch.ones(tokens, tokens, dtype=dtype)

    times, out_msa, out_pair = [], None, None
    for _ in range(passes):
        start = time.perf_counter()
        z = model.extra_msa_stack(extra, pair, extra_mask, pair_mask)
        out_msa, out_pair = model.evoformer_stack(msa, z, msa_mask, pair_mask)
        ttnn.synchronize_device(model._device)
        times.append(time.perf_counter() - start)
    warm = times[1:] or times
    return {"tokens": tokens, "rne_residual": rne, "wide_dram": dram, "passes": passes,
            "first_s": times[0], "warm_s": sum(warm) / len(warm), "warm_min_s": min(warm),
            "_out": (out_msa, out_pair)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="208,512,848")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--arms", default=",".join(a[0] for a in ARMS),
                    help="which arms to run, in order. The L1 OOM depends on what ran "
                         "before it, so dropping an arm is a different experiment")
    ap.add_argument("--params", default=None)
    args = ap.parse_args()

    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    from tap_gate import DEFAULT_PARAMS

    model = load_af2_device_model(load_af2_state_dict(args.params or DEFAULT_PARAMS),
                                  template=False)
    model.eval()

    rows, outs = [], {}
    with torch.no_grad():
        for tokens in (int(t) for t in args.tokens.split(",")):
            for label, rne, dram in (a for a in ARMS if a[0] in args.arms.split(",")):
                try:
                    row = one(model, tokens, args.passes, rne, dram)
                    outs[(tokens, label)] = row.pop("_out")
                except Exception as error:            # OOM is a result, not a crash
                    row = {"tokens": tokens, "rne_residual": rne, "wide_dram": dram,
                           "error": f"{type(error).__name__}: {error}"[:300]}
                rows.append(row | {"arm": label})
                print(json.dumps(rows[-1]), file=sys.stderr, flush=True)

    # Placement is not arithmetic: wherever both wide arms ran, they have to be bit-identical.
    equal = []
    for tokens in sorted({r["tokens"] for r in rows}):
        left, right = outs.get((tokens, "wide_l1")), outs.get((tokens, "wide_dram"))
        if left is None or right is None:
            equal.append({"tokens": tokens, "compared": False,
                          "why": "one wide arm did not run"})
            continue
        equal.append({"tokens": tokens, "compared": True,
                      "msa_bit_exact": bool(torch.equal(left[0], right[0])),
                      "pair_bit_exact": bool(torch.equal(left[1], right[1])),
                      "pair_max_abs": float((left[1] - right[1]).abs().max())})

    cost = []
    for tokens in sorted({r["tokens"] for r in rows}):
        by = {r["arm"]: r for r in rows if r["tokens"] == tokens}
        base = by.get("bf16_add", {}).get("warm_s")
        for label in ("wide_l1", "wide_dram"):
            warm = by.get(label, {}).get("warm_s")
            if base and warm:
                cost.append({"tokens": tokens, "arm": label, "warm_s": warm,
                             "bf16_add_s": base, "ratio": warm / base,
                             "delta_s": warm - base})
    print(json.dumps({"mode": "af2ig_rne_placement", "rows": rows,
                      "bit_exact": equal, "cost": cost}, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
