#!/usr/bin/env python3
"""Is BindCraft 2's gradient round armed with the exact-training instrument? Answered at runtime.

Card-free on purpose. Nothing here opens a device, so it runs on a box whose four cards are all
held, and it answers the half of the question that does not need one: whether the route BindCraft
2 takes arms `softmax` and `layer_norm` on the host float64 path, and whether the counters that
would measure it are wired to the callables that actually get installed.

Asserted from the live process rather than read off the source:

  ARMED     inside `taped_ttnn.tape()` with no `exact_training(...)` anywhere on the stack,
            `ttnn.softmax`, `ttnn.softmax_in_place` and `ttnn.layer_norm` are the host float64
            stand-ins, and `_VERBS` carries the exact verbs. `perf/bcx_predictor/trace_wire.py`
            opens exactly this scope and BindCraft 2's harness never calls `exact_training`.
  DISARMED  the same scope under `exact_training(False)` leaves all six bindings shipped.
  NESTING   an inner `exact_training(True)` beats an outer False, so a harness that wanted the
            instrument off has to own the outermost scope.
  COUNTING  the counters belong to the installed callables. `_exact_softmax_raw` bumps
            EXACT_SOFTMAX_STATS["raw"] before it touches the tensor, so calling the INSTALLED
            binding moves the counter. The argument is a plain torch tensor: it reaches the
            increment and then dies in `ttnn.to_torch`, without going near a device. A ttnn
            HOST tensor is not usable for this -- `ttnn.softmax` on one blocks in the device
            pool, which is how the first version of this probe hung.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                                           # noqa: E402
import ttnn                                                            # noqa: E402

from tt_bio import autograd as ag                                      # noqa: E402
from tt_bio import taped_ttnn as tt                                    # noqa: E402

RAW = ("softmax", "softmax_in_place", "layer_norm")
VERBS = ("softmax", "softmax_in_place", "layer_norm")


#: The exact stand-in for each binding, by identity. A module check is not enough for the
#: taped verbs: the SHIPPED taped `layer_norm` verb also lives in `tt_bio.autograd`, so
#: `__module__` reads exact on it whether the scope is open or not.
EXACT_RAW = {"softmax": "_exact_softmax_raw", "softmax_in_place": "_exact_softmax_raw",
             "layer_norm": "_exact_layer_norm_raw"}
EXACT_VERB = {"softmax": "_v_exact_softmax", "softmax_in_place": "_v_exact_softmax",
              "layer_norm": "_v_exact_layer_norm"}


def _exactness():
    """Which of the six bindings are the exact host stand-ins right now, by identity."""
    raw = {n: getattr(ttnn, n, None) is getattr(ag, EXACT_RAW[n]) for n in RAW}
    verbs = {n: tt._VERBS.get(n) is getattr(ag, EXACT_VERB[n]) for n in VERBS}
    return {"raw": raw, "verbs": verbs,
            "softmax_installed": ag.exact_softmax_installed(),
            "layer_norm_installed": ag.exact_layer_norm_installed()}


def counter_is_on_the_only_path_in():
    before = dict(ag.EXACT_SOFTMAX_STATS)
    with tt.tape():
        fn = ttnn.softmax
        installed = fn is ag._exact_softmax_raw
        try:
            fn(torch.zeros(32, 32), dim=-1)
            raised = None
        except BaseException as exc:                                    # noqa: BLE001
            raised = type(exc).__name__
    after = dict(ag.EXACT_SOFTMAX_STATS)
    return {"binding_was_the_exact_one": installed, "raised": raised,
            "raw_before": before["raw"], "raw_after": after["raw"],
            "counter_moved": after["raw"] - before["raw"] == 1}


def bc2_harness_never_turns_it_off():
    """`git grep` over the harness BindCraft 2 actually runs, on this tree."""
    paths = ["perf/bcx_predictor/", "perf/bcx_round/", "perf/bcx_stack/", "perf/bcx_seam/",
             "perf/bcx_tapedfwd/", "perf/bcx_trace/"]
    here = [p for p in paths if (ROOT / p).exists()]
    out = subprocess.run(["git", "-C", str(ROOT), "grep", "-n", "exact_training", "--"] + here,
                         capture_output=True, text=True)
    off = subprocess.run(["git", "-C", str(ROOT), "grep", "-rn", "exact_training(", "--",
                          "tt_bio/"], capture_output=True, text=True)
    return {"searched": here, "hits_in_bc2_harness": [l for l in out.stdout.splitlines() if l],
            "exact_training_call_sites_in_engine":
                [l for l in off.stdout.splitlines() if l]}


def main():
    blob = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
            "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
            "loadavg": os.getloadavg(), "nproc": os.cpu_count(),
            "opened_a_device": False,
            "EXACT_TRAINING_OPS": list(ag.EXACT_TRAINING_OPS),
            "exact_training_ops()": list(ag.exact_training_ops())
            if hasattr(ag, "exact_training_ops") else None,
            "module_default_EXACT_TRAINING": list(ag._EXACT_TRAINING)}

    blob["before_any_scope"] = _exactness()
    with tt.tape():
        blob["inside_tape_module_default"] = _exactness()
    blob["after_tape"] = _exactness()
    with ag.exact_training(False):
        with tt.tape():
            blob["inside_tape_exact_training_False"] = _exactness()
    with ag.exact_training(False):
        with ag.exact_training(True):
            with tt.tape():
                blob["innermost_wins"] = _exactness()
    blob["counter"] = counter_is_on_the_only_path_in()
    blob["grep"] = bc2_harness_never_turns_it_off()

    armed = blob["inside_tape_module_default"]
    blob["ANSWER"] = {
        "bc2_round_is_armed": all(armed["raw"].values()) and all(armed["verbs"].values()),
        "off_switch_works": not any(blob["inside_tape_exact_training_False"]["raw"].values()),
        "counter_on_the_path": blob["counter"]["counter_moved"],
        "bc2_harness_calls_exact_training": bool(blob["grep"]["hits_in_bc2_harness"])}

    out = HERE / "ARMED.json"
    out.write_text(json.dumps(blob, indent=1))
    print(json.dumps(blob, indent=1))
    print(f"-> {out}")
    return 0 if blob["ANSWER"]["bc2_round_is_armed"] and blob["ANSWER"]["off_switch_works"] else 1


if __name__ == "__main__":
    sys.exit(main())
