#!/usr/bin/env python3
"""The two atom-branch sites under the ones wave 2 has already taken, measured on the step.

`b2z2-step-program-fusion`'s site map prices every line of the diffusion step. Four rows have
taken the top of it: the atom key gather (twice), the AdaLN conditioning norm, and the token
attention epilogue. What is left in the atom branch, per atom layer and six layers per step:

  kv projection            249.5 us   at 23-26 % of the byte roof (b2z2-step-matmul-group)
  nlp_create_qkv_heads     158.8 us   three quarters of it splitting zeros
  query pad                 69.5 us   writing those zeros
  q / g / o projections    122.0 us   same byte roof
  head slice                23.5 us   taking the zeros back off

Two levers, one flag each, measured here against the same base in the same process:

  TT_BIO_ATOM_L1              the branch's live set in L1 instead of a DRAM round trip
  TT_BIO_ATOM_HEADS_UNPADDED  the head split as a permute, so the pad and the slice go away

Both are bit-exact by construction -- one moves bytes between memories, the other permutes the
same channels -- so this harness checks `torch.equal` on the step output for every arm and refuses
to report a ratio for an arm that fails it. A micro-probe over-predicted its own step win by 38 %
on this same block, so nothing here is measured off the fold: the arms replay the real
`Diffusion.__call__` with its shipped shapes, interleaved, profiler off.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_step_fusion"))

ARMS = {
    "base":  {"_ATOM_L1": False, "_ATOM_HEADS_UNPADDED": False},
    "l1":    {"_ATOM_L1": True,  "_ATOM_HEADS_UNPADDED": False},
    "heads": {"_ATOM_L1": False, "_ATOM_HEADS_UNPADDED": True},
    "both":  {"_ATOM_L1": True,  "_ATOM_HEADS_UNPADDED": True},
}


def set_arm(T, arm):
    for k, v in ARMS[arm].items():
        setattr(T, k, v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="base,l1,heads,both")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()
    arms = a.arms.split(",")
    for arm in arms:
        assert arm in ARMS, arm

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import step_probe as SP

    SP.OUT_PATH = a.out
    SP.OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "card": os.environ.get("TT_VISIBLE_DEVICES"),
                     "size": a.size, "arms": arms, "reps": a.reps, "blocks": a.blocks,
                     "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                     "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                     "loadavg": open("/proc/loadavg").read().split()[:3]}
    out = SP.OUT
    a.out.parent.mkdir(parents=True, exist_ok=True)

    set_arm(T, "base")
    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    out["env"]["arch"] = str(dev.arch())
    fence = SP.make_fence(ttnn, dev)

    def run():
        return g["obj"](*g["args"], **g["kwargs"])

    # Parity first, and on the same grabbed operands every arm is about to be timed on.
    set_arm(T, "base")
    ref = ttnn.to_torch(run())
    parity = {}
    for arm in arms:
        set_arm(T, arm)
        got = ttnn.to_torch(run())
        parity[arm] = {"bit_exact": bool(torch.equal(ref, got)),
                       "max_abs": float((ref - got).abs().max()),
                       "atom_l1_stats": dict(T.ATOM_L1_STATS)}
        T.ATOM_L1_STATS.update(l1=0, dram=0)
        print(f"  parity {arm:6s} {parity[arm]}", flush=True)
    out["parity"] = parity
    SP.dump()

    # Warm every arm before any of them is timed, so no arm pays another arm's first-call cost.
    for arm in arms:
        set_arm(T, arm)
        for _ in range(3):
            run()
    ttnn.synchronize_device(dev)

    walls = {arm: [] for arm in arms}
    order = list(arms)
    for b in range(a.blocks):
        fence()
        for arm in (order if b % 2 == 0 else order[::-1]):
            set_arm(T, arm)
            t0 = time.perf_counter()
            for _ in range(a.reps):
                run()
            ttnn.synchronize_device(dev)
            walls[arm].append(1e3 * (time.perf_counter() - t0) / a.reps)
        print(f"  block {b}: " + "  ".join(f"{k}={walls[k][-1]:.4f}" for k in arms), flush=True)
        out["walls"] = {k: [round(x, 4) for x in v] for k, v in walls.items()}
        SP.dump()

    med = {k: st.median(v) for k, v in walls.items()}
    base = med[arms[0]]
    out["ms_per_step"] = {k: round(v, 4) for k, v in med.items()}
    out["spread_pct"] = {k: round(100 * (max(v) - min(v)) / st.median(v), 3)
                         for k, v in walls.items()}
    out["ratio_vs_base"] = {k: round(base / v, 5) for k, v in med.items()}
    SP.dump()
    print("\n  ms/step " + json.dumps(out["ms_per_step"]))
    print("  ratio   " + json.dumps(out["ratio_vs_base"]))
    print("  spread% " + json.dumps(out["spread_pct"]))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
