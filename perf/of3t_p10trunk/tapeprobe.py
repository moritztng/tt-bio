"""Where the taped trunk cycle's seconds go: the op, the tape, or the eviction.

`of3t-p10trunk` reproduced the 230-300 s taped trunk cycle on pc card 0 at S=4 unchunked on
BOTH harness versions (291.438 s new, 298.469 s old) against a 4.168 / 3.983 / 4.015 s
baseline taken on the same box, same card, same commit, same argv at 09:20 the same day. The
harness is innocent and `tt_bio/` is byte-identical, so the split has to come out of the run.

One hook does it. `_tape` runs AFTER the op it records, so the wall clock between one `_tape`
returning and the next one entering is "the op plus the python around it", and the time inside
`_tape` is the tape's own. `_evict_read_parents` is timed inside that, since it is the named
suspect and it is the only part of `_tape` that issues device work.
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
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        import ttnn
        from tt_bio import autograd as ag
        from tt_bio.tenstorrent import get_device

        held, _meta = S.capture(a.tokens, out)
        trunk = held["trunk"][0]
        dev = get_device()
        declare_all(trunk, held["sampler"][0], out)

        st = {"nodes": 0, "taped_nodes": 0, "tape_s": 0.0, "evict_s": 0.0,
              "evict_calls": 0, "evicted": 0, "between_s": 0.0, "last": None}
        real_tape, real_evict, real_evict_m = ag._tape, ag._evict_read_parents, ag.Tensor.evict
        top = []                                     # (seconds, kind) for the worst gaps

        def probe_evict(parents):
            st["evict_calls"] += 1
            t = time.perf_counter()
            r = real_evict(parents)
            st["evict_s"] += time.perf_counter() - t
            return r

        def probe_evict_m(self, *args, **kw):
            st["evicted"] += 1
            return real_evict_m(self, *args, **kw)

        def probe_tape(out_value, parents, make_fn, reads=None):
            t = time.perf_counter()
            if st["last"] is not None:
                gap = t - st["last"]
                st["between_s"] += gap
                top.append(gap)
            r = real_tape(out_value, parents, make_fn, reads)
            st["nodes"] += 1
            if r.node is not None:
                st["taped_nodes"] += 1
            st["last"] = time.perf_counter()
            st["tape_s"] += st["last"] - t
            return r

        # --- untaped prefix, the control that is 3.4 s in every reading -------------------
        t0 = time.perf_counter()
        trunk_forward(trunk, held, a.cycles - 1, taped=False)
        ttnn.synchronize_device(dev)
        out["prefix_untaped_s"] = round(time.perf_counter() - t0, 3)
        print(f"PREFIX {out['prefix_untaped_s']} s", flush=True)
        dump()

        # --- the taped cycle, instrumented ------------------------------------------------
        ag._tape = probe_tape
        ag._evict_read_parents = probe_evict
        ag.Tensor.evict = probe_evict_m
        with ag.tape():
            t0 = time.perf_counter()
            trunk_forward(trunk, held, 1, taped=True)
            ttnn.synchronize_device(dev)
            taped = time.perf_counter() - t0
        ag._tape, ag._evict_read_parents, ag.Tensor.evict = real_tape, real_evict, real_evict_m

        top.sort(reverse=True)
        out["taped_cycle_s"] = round(taped, 3)
        out["split"] = {
            "nodes": st["nodes"], "taped_nodes": st["taped_nodes"],
            "between_tape_calls_s": round(st["between_s"], 3),
            "inside_tape_s": round(st["tape_s"], 3),
            "of_which_evict_s": round(st["evict_s"], 3),
            "evict_calls": st["evict_calls"], "tensors_evicted": st["evicted"],
            "unaccounted_s": round(taped - st["between_s"] - st["tape_s"], 3),
            "mean_gap_ms": round(1000 * st["between_s"] / max(st["nodes"] - 1, 1), 3),
            "worst_10_gaps_s": [round(x, 3) for x in top[:10]],
        }
        print("SPLIT " + json.dumps(out["split"]), flush=True)
        out["dram"] = int(ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank) \
            if hasattr(ttnn, "get_memory_view") else None
        dump()
    out.setdefault("env", {})["aiclk_during"] = clk.summary()
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
