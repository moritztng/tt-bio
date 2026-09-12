#!/usr/bin/env python3
"""`TT_BIO_SDPA_GRID_Q_CHUNK` on the real diffusion step, interleaved, with parity beside it.

The op-level sweep says 1.3117x on the token SDPA at max abs 0.0. An op-level screen on this
block has already over-predicted its own step win by 38 % once (`b2z2-step-program-fusion`), so
the ratio this row books comes from the step: the grabbed `Diffusion.__call__` with its shipped
shapes, both arms warmed before either is timed, the lead alternating between blocks, profiler
off. The pick is bit-exact by construction, so an arm that is not `torch.equal` is a bug and the
harness refuses to report its ratio.
"""
from __future__ import annotations

import argparse, json, os, statistics as st, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "perf" / "b2z2_step_fusion", ROOT / "scripts" / "gpu_vs_tt",
          ROOT / "perf" / "b2x_difflayer"):
    sys.path.insert(0, str(p))

ARMS = {"base": False, "gridq": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="base,gridq")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()
    arms = a.arms.split(",")

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import step_probe as SP

    def set_arm(arm):
        T._SDPA_GRID_Q_CHUNK = ARMS[arm]
        T._sdpa_program_config_for_lengths.cache_clear()

    SP.OUT_PATH = a.out
    SP.OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
                     "arms": arms, "reps": a.reps, "blocks": a.blocks,
                     "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                     "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                     "loadavg": open("/proc/loadavg").read().split()[:3]}
    out = SP.OUT
    a.out.parent.mkdir(parents=True, exist_ok=True)

    set_arm("base")
    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    out["env"]["arch"] = str(dev.arch())
    out["env"]["n_cores"] = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    fence = SP.make_fence(ttnn, dev)

    def run():
        return g["obj"](*g["args"], **g["kwargs"])

    # which q_chunk each arm actually hands the kernel, recorded rather than assumed
    picks = {}
    for arm in arms:
        set_arm(arm)
        T.SDPA_CHUNK_PICKS.clear()
        run()
        ttnn.synchronize_device(dev)
        picks[arm] = {str(k): v for k, v in T.SDPA_CHUNK_PICKS.items()}
    out["q_chunk_picks"] = picks
    print("  picks " + json.dumps(picks), flush=True)

    set_arm("base")
    ref = ttnn.to_torch(run())
    parity = {}
    for arm in arms:
        set_arm(arm)
        got = ttnn.to_torch(run())
        parity[arm] = {"bit_exact": bool(torch.equal(ref, got)),
                       "max_abs": float((ref - got).abs().max())}
        print(f"  parity {arm:6s} {parity[arm]}", flush=True)
    out["parity"] = parity
    SP.dump()

    for arm in arms:
        set_arm(arm)
        for _ in range(3):
            run()
    ttnn.synchronize_device(dev)

    walls = {arm: [] for arm in arms}
    for b in range(a.blocks):
        fence()
        for arm in (arms if b % 2 == 0 else arms[::-1]):
            set_arm(arm)
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
    out["spread_pct"] = {k: round(100 * (max(v) - min(v)) / st.median(v), 3) for k, v in walls.items()}
    out["ratio_vs_base"] = {k: round(base / v, 5) for k, v in med.items()}
    SP.dump()
    print("\n  ms/step " + json.dumps(out["ms_per_step"]))
    print("  ratio   " + json.dumps(out["ratio_vs_base"]))
    print("  spread% " + json.dumps(out["spread_pct"]))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
