"""Which verb eats the taped trunk cycle, and whether the eviction is what makes it eat it.

`tapeprobe.py` put all 257.233 s of a 257.313 s taped cycle BETWEEN `_tape` calls: the tape's
own bookkeeping is 0.067 s and `_evict_read_parents` is 0.000 s over 147 calls. So the ops are
slow, not the tape. This run asks which ops, and answers the eviction question by A/B rather
than by the cost of the call: an eviction is cheap to MAKE and expensive to have made, because
every later read of that operand comes from DRAM.

Three arms over one capture, each one trunk cycle, each instrumented identically:
  A  untaped      -- the control that reads 1.14 s/cycle in every artifact this campaign owns
  B  taped        -- the 230-300 s cell
  C  taped, `Tensor.evict` neutered to a no-op -- B minus the only device work the tape adds
"""
from __future__ import annotations

import argparse
import collections
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
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "arms": {}}
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
        real_tape, real_evict_m = ag._tape, ag.Tensor.evict

        def run(name, taped, evict):
            by = collections.defaultdict(lambda: [0, 0.0])        # verb -> [calls, gap s]
            st = {"n": 0, "taped": 0, "tape_s": 0.0, "last": None, "evicted": 0}

            def probe_tape(out_value, parents, make_fn, reads=None):
                t = time.perf_counter()
                if st["last"] is not None:
                    # The caller of `_tape` is the verb that just ran the op.
                    e = by[sys._getframe(1).f_code.co_name]
                    e[0] += 1
                    e[1] += t - st["last"]
                r = real_tape(out_value, parents, make_fn, reads)
                st["n"] += 1
                st["taped"] += r.node is not None
                st["last"] = time.perf_counter()
                st["tape_s"] += st["last"] - t
                return r

            def probe_evict_m(self, *args, **kw):
                st["evicted"] += 1
                return None if not evict else real_evict_m(self, *args, **kw)

            ag._tape, ag.Tensor.evict = probe_tape, probe_evict_m
            t0 = time.perf_counter()
            try:
                with (ag.tape() if taped else ag.no_grad()):
                    trunk_forward(trunk, held, 1, taped=taped)
                ttnn.synchronize_device(dev)
                err = None
            except Exception as exc:                                  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                ag._tape, ag.Tensor.evict = real_tape, real_evict_m
            cyc = time.perf_counter() - t0
            rows = sorted(by.items(), key=lambda kv: -kv[1][1])
            out["arms"][name] = {
                "taped": taped, "evict_enabled": evict, "cycle_s": round(cyc, 3),
                "error": err, "nodes": st["n"], "taped_nodes": st["taped"],
                "evict_calls": st["evicted"], "inside_tape_s": round(st["tape_s"], 3),
                "dram_after": int(ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
                                  .total_bytes_allocated_per_bank) * 12,
                "by_verb": [{"verb": v, "calls": c, "gap_s": round(s, 3),
                             "ms_per_call": round(1000 * s / c, 3)}
                            for v, (c, s) in rows[:14]],
            }
            print(f"ARM {name}: cycle {cyc:.3f} s, nodes {st['n']}, taped {st['taped']}, "
                  f"evict_calls {st['evicted']}, err {err}", flush=True)
            for r in out["arms"][name]["by_verb"][:8]:
                print(f"   {r['verb']:<28} {r['calls']:>6} calls  {r['gap_s']:>9.3f} s"
                      f"  {r['ms_per_call']:>8.3f} ms/call", flush=True)
            dump()

        plan = {"A": ("A", False, True), "B": ("B", True, True), "C": ("C", True, False)}
        for k in a.arms.split(","):
            run(*plan[k.strip()])

    out.setdefault("env", {})["aiclk_during"] = clk.summary()
    print("CLOCK " + json.dumps(out["env"]["aiclk_during"]), flush=True)
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
