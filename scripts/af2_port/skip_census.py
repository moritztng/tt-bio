"""Where inside the AF2 trunk do the seconds go, measured by dropping one op class at a time.

The trunk at 848 tokens costs far more per Evoformer block than tt-bio's own Protenix pairformer
does at the same dims, and a per-op screen cannot say why: it oversyncs by ~2x
(`tt-bio-isolated-op-timing-oversync-inflates-cost`) and it prices an op against its own captured
input rather than in the chain. This leaves every op in the chain and removes one class per leg, so
a class's share is `incumbent - leg`.

`AF2PairBlock.skip` is the instrument, not `substitute`: substitution moves an op to host torch, so
a leg would read `host_X - device_X`. A skipped op does not run and its residual add becomes the
identity, which means a class's share here is the op plus the one add that carries it. The residual
adds are small on this trunk (0.42 s over four passes, `AF2PairBlock.rne_residual`) but they are not
zero, so the attribution is stated rather than hidden.

Every leg runs in ONE process so the ttnn compile is paid once and every leg is warm. The legs are
interleaved and the incumbent runs first and last as its own A/A control: if those two disagree by
more than the class shares being read, the box was not quiet and nothing here is a measurement.

Arithmetically wrong on purpose, on synthetic inputs. No accuracy claim.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=. \\
        env/bin/python3 scripts/af2_port/skip_census.py --tokens 848 --passes 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

NUM_MSA_ROWS = 2


def inputs(tokens: int, dtype) -> dict:
    """`trunk_timing.py`'s synthetic tensors, at the same seed, so the two harnesses agree."""
    from tt_bio.af2_reference import C_EXTRA, C_M, C_Z

    generator = torch.Generator().manual_seed(0)
    return {
        "msa": torch.randn(NUM_MSA_ROWS, tokens, C_M, generator=generator).to(dtype),
        "pair": torch.randn(tokens, tokens, C_Z, generator=generator).to(dtype),
        "extra": torch.randn(1, tokens, C_EXTRA, generator=generator).to(dtype),
        "extra_mask": torch.zeros(1, tokens, dtype=dtype),
        "msa_mask": torch.ones(NUM_MSA_ROWS, tokens, dtype=dtype),
        "pair_mask": torch.ones(tokens, tokens, dtype=dtype),
    }


def one_pass(model, t: dict) -> float:
    """One trunk pass: both stacks, one timing boundary. Each stack ends in a blocking
    `to_torch`, so the pass is synchronised without an extra `synchronize_device`."""
    start = time.perf_counter()
    z = model.extra_msa_stack(t["extra"], t["pair"], t["extra_mask"], t["pair_mask"])
    model.evoformer_stack(t["msa"], z, t["msa_mask"], t["pair_mask"])
    return time.perf_counter() - start


def leg(model, t: dict, names, passes: int) -> dict:
    model.set_skip(names)
    times = [one_pass(model, t) for _ in range(passes)]
    warm = times[1:] or times
    return {"passes": passes, "first_s": times[0], "warm_median_s": statistics.median(warm),
            "warm_min_s": min(warm), "all_s": times}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=848)
    ap.add_argument("--passes", type=int, default=3, help="per leg; the first is discarded")
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.af2 import SUBSTITUTION_CLASSES, load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    from tap_gate import DEFAULT_PARAMS

    model = load_af2_device_model(load_af2_state_dict(args.params or DEFAULT_PARAMS),
                                  template=False)
    model.eval()
    t = inputs(args.tokens, model.trunk_dtype)

    classes = [c for c in ("trimul", "triatt", "msa_row", "msa_col", "transitions", "opm")]
    order = ["none"] + classes + ["all", "none"]        # incumbent first and last: the A/A control
    rows = []
    with torch.no_grad():
        for index, name in enumerate(order):
            names = () if name == "none" else SUBSTITUTION_CLASSES[name]
            row = {"leg": index, "skip": name, **leg(model, t, names, args.passes)}
            rows.append(row)
            print(json.dumps(row), file=sys.stderr, flush=True)
    model.set_skip(())

    incumbents = [r["warm_median_s"] for r in rows if r["skip"] == "none"]
    base = statistics.median(incumbents)
    aa_pct = 100 * abs(incumbents[-1] - incumbents[0]) / base
    shares = {r["skip"]: {"leg_s": r["warm_median_s"],
                          "share_s": base - r["warm_median_s"],
                          "share_pct": 100 * (base - r["warm_median_s"]) / base}
              for r in rows if r["skip"] not in ("none",)}
    named = [k for k in classes]
    report = {
        "mode": "af2ig_skip_census",
        "tokens": args.tokens,
        "passes_per_leg": args.passes,
        "incumbent_warm_median_s": base,
        "incumbent_aa_delta_pct": aa_pct,
        "shares": shares,
        # The six classes are every op in an Evoformer block. What they do not cover is the
        # residual adds of the ops that stay, the layer norms folded into each op, and the
        # up/download at each stack boundary.
        "named_share_sum_s": sum(shares[k]["share_s"] for k in named),
        "named_share_sum_pct": 100 * sum(shares[k]["share_s"] for k in named) / base,
        "all_skipped_leg_s": shares["all"]["leg_s"],
        "unattributed_s": shares["all"]["leg_s"],
        "rows": rows,
    }
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
