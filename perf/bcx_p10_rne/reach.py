#!/usr/bin/env python3
"""Does the fold REACH, and where do the residual's operands actually sit?

Two questions the VJP grade cannot answer, both about the same four calls:

1. **Reach.** `rne_fold_cast` is graded bit-identical on the VJP, which is exactly what a lever
   that never ran also reads. So count the residual's OWN ttnn calls: `ttnn.typecast` and
   `ttnn.add` are swapped for counting proxies for the duration of each `_residual` call and put
   back after, which counts that residual and nothing else. Off the fold the residual is
   3 typecasts; on it, 2.

2. **Leg 3, `rne_wide_dram`.** `af2.py:326` forces the two f32 temporaries to DRAM because at
   512 tokens the f32 pair is 128 MB and does not fit L1. At 288 it is 42.5 MB and the sizing
   argument may not hold -- but the flag only does anything if the residual's INPUT is in L1,
   because `wide_config` falls back to `x.memory_config()`. So record the buffer type of `x`.
   If the taped round hands the residual a DRAM tensor, the flag is inert at 288 and there is
   nothing to gate.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_p10_tmplemb"))

import ttnn                                                        # noqa: E402
from vjp import DEFAULT_PARAMS, aiclk, device_arm                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--rne-fold", dest="rne_fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.af2 import AF2PairBlock, load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    AF2PairBlock.rne_fold_cast = bool(args.rne_fold)

    calls = collections.Counter()
    where = collections.Counter()
    dtypes = collections.Counter()
    shipped = AF2PairBlock._residual

    def counted(name, fn):
        def proxy(*a, **kw):
            calls[name] += 1
            if name == "add" and "dtype" in kw:
                dtypes["add dtype=%s" % kw["dtype"]] += 1
            return fn(*a, **kw)
        return proxy

    def residual(self, x, update):
        if update is not None:
            where["x %s %s" % (x.memory_config().buffer_type, x.dtype)] += 1
            calls["residual"] += 1
        tc, add = ttnn.typecast, ttnn.add
        ttnn.typecast, ttnn.add = counted("typecast", tc), counted("add", add)
        try:
            return shipped(self, x, update)
        finally:
            ttnn.typecast, ttnn.add = tc, add

    AF2PairBlock._residual = residual

    state = load_af2_state_dict(args.params, multimer=True)
    model = load_af2_device_model(state, template=True, multimer=True, structure=False,
                                  trunk_dtype=torch.bfloat16).eval()
    g = torch.Generator().manual_seed(0)
    c_t = model.template.output_norm.weight.shape[0]
    act = torch.randn(args.n, args.n, c_t, generator=g)
    mask = torch.ones(args.n, args.n)
    cot = torch.randn(args.n, args.n, c_t, generator=g)
    clk = aiclk()
    device_arm(model, act, mask, cot, 1, "template")

    n_res = calls["residual"]
    report = {
        "n": args.n, "rne_fold": bool(args.rne_fold), "aiclk_before": clk,
        "aiclk_after": aiclk(),
        "residual_calls": n_res,
        "calls_in_residual": dict(calls),
        "per_residual": {k: round(v / max(n_res, 1), 3)
                         for k, v in calls.items() if k != "residual"},
        "add_dtype_kwarg": dict(dtypes),
        "operand_placement": dict(where),
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"reach_fold{args.rne_fold}_n{args.n}.json").write_text(
        json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
