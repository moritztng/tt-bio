#!/usr/bin/env python3
"""E6 follow-up: is the chunked-Transition loop matmul-bound or dispatch-bound?

One process, two folds, the tuned config flipped between them. Per fold it records:
  module_s  -- Transition.__call__ elapsed with synchronize_device on both sides, so it is real
               device+host time for the module and not queue time
  issue_s   -- host time to return from the three ttnn.linear calls, no sync: pure dispatch
If issue_s is a large fraction of module_s, the loop is dispatch-bound and device-side matmul wins
cannot reach the wall. That is the hypothesis; this measures it.
"""
import json
import sys
import time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/perfwar-chunked-transition-cb")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "gpu_vs_tt"))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "protenix-v2"
    import ttnn
    from tt_bio import tenstorrent as T

    state = {"on": True, "module_s": 0.0, "issue_s": 0.0, "calls": 0, "mod_calls": 0}
    orig_linear = T._transition_linear
    orig_call = T.Transition.__call__

    def linear(x, w, ckc, dtype, memory_config, activation=None):
        t = time.perf_counter()
        if state["on"]:
            y = orig_linear(x, w, ckc, dtype, memory_config, activation=activation)
        else:
            y = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=dtype,
                            memory_config=memory_config, core_grid=T.CORE_GRID_MAIN,
                            activation=activation)
        state["issue_s"] += time.perf_counter() - t
        state["calls"] += 1
        return y

    def call(self, x):
        dev = x.device()
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        y = orig_call(self, x)
        ttnn.synchronize_device(dev)
        state["module_s"] += time.perf_counter() - t
        state["mod_calls"] += 1
        return y

    T._transition_linear = linear
    T.Transition.__call__ = call

    import tt_baseline as B
    B.RECYCLING_STEPS = 1
    B.SAMPLING_STEPS = 8
    one_fold, meta, st = B.build_fold(
        model, Path("/tmp/e6-msa"), WT / "examples" / "prot300.yaml",
        WT / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m")
    one_fold()  # cold, discarded
    out = {}
    for arm in ("on", "off", "on", "off"):
        state.update(on=(arm == "on"), module_s=0.0, issue_s=0.0, calls=0, mod_calls=0)
        t0 = time.perf_counter()
        _s, m = one_fold()
        r = dict(fold_s=round(time.perf_counter() - t0, 3),
                 module_s=round(state["module_s"], 3), issue_s=round(state["issue_s"], 3),
                 linear_calls=state["calls"], module_calls=state["mod_calls"],
                 plddt=m.get("plddt"))
        out.setdefault(arm, []).append(r)
        print(arm, json.dumps(r), flush=True)
    (WT / "perf" / "chunked_transition" / f"dispatch_{model}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    st.reset()
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())
