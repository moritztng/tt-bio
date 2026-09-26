"""The taped trunk cycle with the float64 host levers on and off, in one process.

`of3t-p10axis`'s 4.168 / 3.983 / 4.015 s baseline records `"exact_training": false` in its own
arm block. Every 228-301 s reading in this campaign ran the DEFAULT, which is true. `tape()`
installs `EXACT_TRAINING_OPS` = softmax + layer_norm, each a float64 round trip to the host, and
`tapeprobe2` put 254.8 s of a 255.8 s taped cycle in exactly those two verbs. This run is the
A/B that settles it: same process, same capture, same card, back to back.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402
from perf.of3t_stepfloor.fullstep import declare_all, trunk_forward   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "arms": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    def avail():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / (1024 ** 2), 3)

    with during() as clk:
        import ttnn
        from tt_bio import autograd as ag
        from tt_bio.tenstorrent import HOST_F64_SOFTMAX_STATS as SM
        from tt_bio.tenstorrent import get_device

        held, _meta = S.capture(a.tokens, out)
        trunk = held["trunk"][0]
        dev = get_device()
        declare_all(trunk, held["sampler"][0], out)

        def cycle(exact_on, rep):
            s0, l0 = dict(SM), dict(ag.EXACT_LAYER_NORM_STATS)
            x0 = dict(ag.EXACT_SOFTMAX_STATS)
            t = time.perf_counter()
            with ag.exact_training(exact_on):
                with ag.tape():
                    ops = ag.exact_training_ops()
                    inst = (ag.exact_softmax_installed(), ag.exact_layer_norm_installed())
                    trunk_forward(trunk, held, 1, taped=True)
                    ttnn.synchronize_device(dev)
            s = round(time.perf_counter() - t, 3)
            row = {"rep": rep, "exact_training": exact_on, "taped_cycle_s": s,
                   "exact_training_ops": list(ops),
                   "exact_softmax_installed": inst[0], "exact_layer_norm_installed": inst[1],
                   "host_f64_softmax_served": SM["served"] - s0["served"],
                   "exact_layer_norm_verb": (ag.EXACT_LAYER_NORM_STATS["verb"] - l0["verb"]),
                   "EXACT_SOFTMAX_STATS": {k: ag.EXACT_SOFTMAX_STATS[k] - x0.get(k, 0)
                                           for k in ag.EXACT_SOFTMAX_STATS},
                   "EXACT_LAYER_NORM_STATS": {k: ag.EXACT_LAYER_NORM_STATS[k] - l0.get(k, 0)
                                              for k in ag.EXACT_LAYER_NORM_STATS},
                   "HOST_F64_SOFTMAX_STATS": {k: SM[k] - s0.get(k, 0) for k in SM},
                   "host_avail_gib": avail(),
                   "dram": int(ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
                               .total_bytes_allocated_per_bank) * 12}
            out["arms"].append(row)
            print("ARM " + json.dumps(row), flush=True)
            dump()

        # Untaped control first: 3.41-3.47 s of prefix in every artifact this campaign owns.
        t = time.perf_counter()
        trunk_forward(trunk, held, 3, taped=False)
        ttnn.synchronize_device(dev)
        out["prefix_untaped_3cycles_s"] = round(time.perf_counter() - t, 3)
        print(f"PREFIX {out['prefix_untaped_3cycles_s']} s", flush=True)
        dump()

        for rep in range(a.reps):
            cycle(False, rep)
        for rep in range(a.reps):
            cycle(True, rep)

    out.setdefault("env", {})["aiclk_during"] = clk.summary()
    print("CLOCK " + json.dumps(out["env"]["aiclk_during"]), flush=True)
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
