"""Which `ops.taping()` gate makes the taped and untaped Evoformer block compute different things?

Every op the taped block runs is bit-identical to its shipped verb on the same operands
(`op_shadow_diff.py`), and the two arms still run different op sequences (`op_trace_diff.py`).
`tenstorrent.py` asks `ops.taping()` at a dozen places to choose a different program or
placement under a tape -- an L1-out projection config, an in-place residual, a DRAM memory
config. Those gates are the mechanism with exactly the measured shape.

So this forces them from the outside. `ops.taping` is replaced by a function that answers the
real value, OR True when the asking line is in the forced set. The taped arm is untouched (its
real answer is already True); the untaped arm takes the taped program at the forced gates only.
Forcing all of them and seeing the block difference go to zero puts the whole effect inside the
gates; forcing one at a time then names the gate.

  TT_VISIBLE_DEVICES=<card> python3 perf/bci_accept/taping_gates.py --card <card> \
      --af2-weights ~/bcx_e2e/af2_params --tokens 64
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np


def rel_l2(a, b):
    d = float(np.linalg.norm(np.asarray(b).ravel()))
    return float("nan") if d == 0 else float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--model", default="model_1_multimer_v3")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rows", type=int, default=3, help="blocks measured per configuration")
    parser.add_argument("--only-all", action="store_true", help="control and ALL legs only")
    args = parser.parse_args()
    if "TT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set TT_VISIBLE_DEVICES to the leased card before running this")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(device=args.card)

    from tt_bio import bindcraft2, ops, tenstorrent

    source = open(tenstorrent.__file__).read().splitlines()
    gates = [i + 1 for i, line in enumerate(source) if re.search(r"\btaping\(\)", line)
             and not line.lstrip().startswith("#")]
    print(f"{len(gates)} taping() gates in {tenstorrent.__file__}: {gates}", flush=True)

    real_taping = ops.taping
    forced: set = set()
    asked: dict = {}

    def taping():
        frame = sys._getframe(1)
        if frame.f_code.co_filename == tenstorrent.__file__:
            real = real_taping()
            # Split by arm: a gate the UNTAPED arm never asks cannot be forced from here, and
            # an all-digits-identical leg would then be an inert lever, not a null.
            key = (frame.f_lineno, "taped" if real else "untaped")
            asked[key] = asked.get(key, 0) + 1
            if frame.f_lineno in forced:
                return True
            return real
        return real_taping()

    ops.taping = taping

    rng = np.random.default_rng(args.seed)
    n = args.tokens
    msa = (rng.standard_normal((1, n, 256)) * args.scale).astype(np.float32)
    pair = (rng.standard_normal((n, n, 128)) * args.scale).astype(np.float32)
    mask = np.ones((1, n), dtype=np.float32)
    pair_mask = np.ones((n, n), dtype=np.float32)

    pool = bindcraft2.TrunkPool(args.af2_weights)
    pool.require([args.model])
    pool.use(args.model)

    with bindcraft2.refusals_unwrapped(), bindcraft2.fast_round():
        evo = bindcraft2.EvoformerOnDevice(pool, blocks=bindcraft2.EVOFORMER_BLOCKS,
                                           recompute=False, memory=bindcraft2._Memory("fast"))
        trunk = evo._trunk("")
        blocks = list(trunk.model.device_evoformer)
        try:
            carry_msa, carry_pair = msa, pair
            for index in range(args.warmup):
                trunk.model.device_evoformer = [blocks[index]]
                carry_msa, carry_pair = evo._primal("", carry_msa, carry_pair, mask, pair_mask)
            start = (carry_msa, carry_pair)

            def measure(label, lines):
                forced.clear()
                forced.update(lines)
                asked.clear()
                c_msa, c_pair = start
                readings = []
                try:
                    for index in range(args.warmup, args.warmup + args.rows):
                        trunk.model.device_evoformer = [blocks[index]]
                        untaped = evo._primal("", c_msa, c_pair, mask, pair_mask)
                        taped = evo._taped("", c_msa, c_pair, mask, pair_mask)
                        readings.append((rel_l2(taped[0], untaped[0]), rel_l2(taped[1], untaped[1])))
                        c_msa, c_pair = untaped
                except Exception as error:                      # a forced gate may not run raw
                    print(f"{label:>34}  FAILED {type(error).__name__}: {str(error)[:120]}",
                          flush=True)
                    return
                msa_mean = float(np.mean([r[0] for r in readings]))
                pair_mean = float(np.mean([r[1] for r in readings]))
                reached = sorted(f"{k[0]}:{k[1]}x{v}" for k, v in asked.items()
                                 if not lines or k[0] in lines)
                print(f"{label:>34}  msa {msa_mean:.6f}  pair {pair_mean:.6f}  "
                      f"gates asked {reached}", flush=True)

            print(f"\n{'forced in the untaped arm':>34}  mean over {args.rows} blocks", flush=True)
            measure("(none: control)", set())
            measure("ALL gates", set(gates))
            if not args.only_all:
                for line in gates:
                    measure(f"tenstorrent.py:{line}", {line})
        finally:
            ops.taping = real_taping
            trunk.model.device_evoformer = blocks


if __name__ == "__main__":
    main()
