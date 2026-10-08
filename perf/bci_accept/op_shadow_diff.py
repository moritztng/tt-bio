"""Which taped op computes a different forward from the shipped one, named by op and call site.

The skip sweep narrowed the pair-track taped/untaped difference to `tri_att_start` and
`tri_att_end` -- skip both and the pair track reads exactly 0.000000 -- and then ran out of
levers. No kernel set, no `mm_layout`, no softmax compute-kernel config and no generic wrapper
accounts for what is left, so the next step cannot be another flag.

This asks the tape directly. Every entry in `taped_ttnn._VERBS` is wrapped so that, after it
computes its taped value, the SAME shipped verb is called a second time on the SAME raw
operands and the two are compared. A verb whose taped value matches its shipped one to every
bit contributes nothing by construction; a verb that differs is a carrier, and the wrapper
records which tt-bio line called it.

This is per-op INJECTION, not accumulation: both arms of each comparison are fed the identical
input the taped run actually had, so a large reading cannot be inherited from an op upstream.
That is the property the block-level sweep did not have -- there, skipping an op also removed
everything it fed.

In-place verbs are skipped rather than measured. Calling `softmax_in_place` or anything whose
name ends in `_` a second time would apply it twice to the same buffer, which is a wrong
number and a corrupted run, not a measurement.

  TT_VISIBLE_DEVICES=<card> python3 perf/bci_accept/op_shadow_diff.py --card <card> \
      --af2-weights ~/bcx_e2e/af2_params --tokens 64
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

#: Verbs that mutate an operand, so calling the shipped one a second time on the same buffer
#: changes the number instead of measuring it.
IN_PLACE = {"softmax_in_place", "deallocate", "reallocate"}


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b.ravel()))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((a - b).ravel()) / denominator)


def caller_site() -> str:
    """The first tt-bio frame outside the tape, which names the module that issued the op."""
    frame = sys._getframe(1)
    while frame is not None:
        path = frame.f_code.co_filename
        if "/tt_bio/" in path and not path.endswith(("/taped_ttnn.py", "/autograd.py",
                                                     "/op_shadow_diff.py")):
            return f"{path.rsplit('/tt_bio/', 1)[1]}:{frame.f_lineno}"
        frame = frame.f_back
    return "<outside tt_bio>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--model", default="model_1_multimer_v3")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--n-seq", type=int, default=1)
    parser.add_argument("--c-m", type=int, default=256)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2,
                        help="blocks the carry is advanced untaped before the measured block. "
                             "A raw random input is off-distribution and the first block "
                             "amplifies it, which is what inflated this row's earlier numbers.")
    parser.add_argument("--top", type=int, default=40, help="rows printed")
    args = parser.parse_args()

    if "TT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set TT_VISIBLE_DEVICES to the leased card before running this")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(device=args.card)

    import torch
    import ttnn
    from tt_bio import bindcraft2, taped_ttnn
    from tt_bio.autograd import _raw

    #: (op, site) -> [calls, differing, sum rel L2, max rel L2, errors]
    stats: dict = {}

    def to_numpy(t):
        return ttnn.to_torch(t).to(dtype=torch.float32).numpy()

    def shadow(name, entry):
        def wrapped(shipped, call_args, call_kwargs):
            out = entry(shipped, call_args, call_kwargs)
            if not measuring[0] or out is None:
                return out
            value = getattr(out, "value", None)
            if not isinstance(value, ttnn.Tensor):
                return out
            key = (name, caller_site())
            row = stats.setdefault(key, [0, 0, 0.0, 0.0, 0])
            row[0] += 1
            reference = None
            try:
                raw_args, raw_kwargs = _raw(call_args, call_kwargs)
                measuring[0] = False          # the shadow call must not re-enter this wrapper
                reference = shipped(*raw_args, **raw_kwargs)
                if not isinstance(reference, ttnn.Tensor):
                    return out
                taped_values, shipped_values = to_numpy(value), to_numpy(reference)
                if taped_values.shape != shipped_values.shape:
                    row[4] += 1
                    return out
                difference = rel_l2(taped_values, shipped_values)
                row[2] += difference
                row[3] = max(row[3], difference)
                if difference != 0.0:
                    row[1] += 1
            except Exception:                 # an op the shadow cannot replay is counted, not fatal
                row[4] += 1
            finally:
                # The reference is dropped, never deallocated. Several shipped verbs --
                # reshape, typecast, to_layout -- hand back a view onto an operand's own
                # buffer, so `ttnn.deallocate` on the result frees an INPUT that the run is
                # still using: the first attempt died one op later in `_k_rne_add` with
                # "Tensor is not allocated". Letting the binding's refcount free it is both
                # correct and enough at this token axis.
                measuring[0] = True
            return out
        return wrapped

    measuring = [False]

    rng = np.random.default_rng(args.seed)
    n = args.tokens
    msa = (rng.standard_normal((args.n_seq, n, args.c_m)) * args.scale).astype(np.float32)
    pair = (rng.standard_normal((n, n, args.c_z)) * args.scale).astype(np.float32)
    mask = np.ones((args.n_seq, n), dtype=np.float32)
    pair_mask = np.ones((n, n), dtype=np.float32)

    pool = bindcraft2.TrunkPool(args.af2_weights)
    pool.require([args.model])
    if args.model in pool.absent:
        raise SystemExit(f"{args.model} not in {args.af2_weights}: {pool.absent[args.model]}")
    pool.use(args.model)

    print(f"tokens {n}, seed {args.seed}, warmup {args.warmup}", flush=True)
    print(f"verbs wrapped: {len(taped_ttnn._VERBS)}, skipped as in-place: "
          f"{sorted(IN_PLACE & set(taped_ttnn._VERBS))}", flush=True)

    with bindcraft2.refusals_unwrapped(), bindcraft2.fast_round():
        evo = bindcraft2.EvoformerOnDevice(pool, blocks=bindcraft2.EVOFORMER_BLOCKS,
                                           recompute=False,
                                           memory=bindcraft2._Memory("fast"))
        trunk = evo._trunk("")
        blocks = list(trunk.model.device_evoformer)
        try:
            carry_msa, carry_pair = msa, pair
            for index in range(args.warmup):
                trunk.model.device_evoformer = [blocks[index]]
                carry_msa, carry_pair = evo._primal("", carry_msa, carry_pair, mask, pair_mask)
            print(f"carry settled through {args.warmup} untaped blocks, pair max "
                  f"{float(np.max(np.abs(carry_pair))):.6g}", flush=True)

            for name, entry in list(taped_ttnn._VERBS.items()):
                if name.rsplit(".", 1)[-1].endswith("_") or name in IN_PLACE:
                    continue
                taped_ttnn._VERBS[name] = shadow(name, entry)

            trunk.model.device_evoformer = [blocks[args.warmup]]
            # The untaped arm first, so the block-level total this table has to explain is
            # printed beside the per-op rows rather than quoted from an earlier run.
            untaped = evo._primal("", carry_msa, carry_pair, mask, pair_mask)
            measuring[0] = True
            taped_msa, taped_pair, _token = evo._taped("", carry_msa, carry_pair, mask, pair_mask)
            measuring[0] = False
            print(f"block {args.warmup}: msa rel L2 {rel_l2(taped_msa, untaped[0]):.6f}, "
                  f"pair rel L2 {rel_l2(taped_pair, untaped[1]):.6f}", flush=True)
        finally:
            trunk.model.device_evoformer = blocks

    print(f"\n{'op':>44} {'site':>34} {'calls':>6} {'differ':>7} {'mean relL2':>12} "
          f"{'max relL2':>12} {'err':>4}", flush=True)
    rows = sorted(stats.items(), key=lambda kv: -kv[1][3])
    for (name, site), (calls, differing, total, worst, errors) in rows[:args.top]:
        mean = total / calls if calls else float("nan")
        print(f"{name:>44} {site:>34} {calls:>6} {differing:>7} {mean:>12.6f} "
              f"{worst:>12.6f} {errors:>4}", flush=True)
    carriers = [(k, v) for k, v in stats.items() if v[1]]
    print(f"\n{len(carriers)} of {len(stats)} (op, site) pairs differ from their shipped verb at "
          f"all; the rest are bit-identical and cannot be the carrier.", flush=True)


if __name__ == "__main__":
    main()
