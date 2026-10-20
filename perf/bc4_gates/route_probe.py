#!/usr/bin/env python3
"""Which fused-kernel gates decline under BindCraft 2's open tape, and how many of those calls had no
operand on it.

Wraps `ops.declines_under_tape` and runs the script after `--` (perf/spd/bc2_census.py). Every gate
call is counted per iteration by name: `taped` (a tape is open and the kernel has no entry), `raw`
(of those, no operand on the tape: the call a gate that passes its operands now serves) and
`declined` (what the gate answered). BC4_GATES_ARM=base drops the operands, which is the gate as it
was on main: blind to its operands, so a raw call declines too, except the two permute gates main
already fixed, and the gates main asked with a bare `ops.taping()` answer exactly that.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=. python perf/bc4_gates/route_probe.py out/route.json -- \\
        perf/spd/bc2_census.py --out out/cen --at 4

bc2_census.py ends its process with `os._exit` inside the instrumented iteration, so the record is
written from a wrapper around `os._exit`, not from a `finally`.
"""
from __future__ import annotations

import collections
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

out = Path(sys.argv[1])
assert sys.argv[2] == "--"
argv = sys.argv[3:]

from tt_bio import ops, bindcraft2 as bc2  # noqa: E402

ARM = os.environ.get("BC4_GATES_ARM", "fix")
#: Gates main already asks with their operands, and gates main asked with a bare `ops.taping()`.
MAIN_OPERANDS = {"reblock_permute", "reblock_permute_back"}
MAIN_TAPING = {"gate_proj", "qkvg_heads", "out_proj", "gated_out_proj", "qkvgb_heads", "triatt_sdpa",
               "triatt_sdpa_fused_qkv", "widen_add", "reblock_permute_gated_rowblock"}
BC2 = Path(argv[argv.index("--bc2") + 1] if "--bc2" in argv else "~/bcx_e2e/bc2").expanduser()
sys.path.insert(0, str(BC2))
it = [0]
counts = collections.defaultdict(lambda: collections.Counter())
decl0 = ops.declines_under_tape


def declines(name, *operands):
    taped = ops.taping() and ops._KERNEL_ENTRIES.get(name) is None
    if ARM != "base" or name in MAIN_OPERANDS:
        res = decl0(name, *operands)
    else:
        res = ops.taping() if name in MAIN_TAPING else decl0(name)
    c = counts[it[0]]
    c[(name, "calls")] += 1
    if taped:
        c[(name, "taped")] += 1
        on = ops.grad_hook().on_tape(operands, {}) if operands else True
        c[(name, "on_tape" if on else "raw")] += 1
        c[(name, "recording" if ops.recording() else "not_recording")] += 1
    if res:
        c[(name, "declined")] += 1
    return res


ops.declines_under_tape = declines

cls = bc2.design_model_class()
grads0 = cls.sequence_gradients


def sequence_gradients(self, *a, **k):
    it[0] += 1
    return grads0(self, *a, **k)


cls.sequence_gradients = sequence_gradients


def write():
    per_it = {}
    for i, c in sorted(counts.items()):
        d = collections.defaultdict(dict)
        for (name, what), v in c.items():
            d[name][what] = v
        per_it[str(i)] = dict(sorted(d.items()))
    out.write_text(json.dumps(dict(argv=argv, arm=ARM, iterations=it[0], per_iteration=per_it), indent=1))


exit0 = os._exit


def _exit(code):
    write()
    exit0(code)


os._exit = _exit
sys.argv = [str(ROOT / argv[0])] + argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    write()
