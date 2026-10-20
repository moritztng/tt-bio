#!/usr/bin/env python3
"""Which kernel each trimul channel move takes inside BindCraft 2's taped stacks, and why.

Runs `perf/spd/bc2_evo_devprof.py` (its argv after `--`) with the reblock gates and the taped
permute shim instrumented, then writes the gate rejections and every taped permute's route:

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=. python perf/bc4_permute/route_probe.py out/route.json -- \
        --params <multimer_v3.npz> --out out/route --warm 1 --reps 1
"""
from __future__ import annotations

import collections
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

out = Path(sys.argv[1])
assert sys.argv[2] == "--"
argv = sys.argv[3:]

from tt_bio import reblock_permute as R, taped_ttnn as TT, tenstorrent as T  # noqa: E402

routes = collections.Counter()


def _sig(x):
    mc = x.memory_config()
    return f"{[int(d) for d in x.shape]}:{mc.buffer_type}:{mc.memory_layout}"


fwd0 = TT._permute_fwd


def permute_fwd(shipped, ra, rk, dims):
    before = (R.STATS[0], R.STATS_BACK[0], R.STATS_PBPK[0])
    res = fwd0(shipped, ra, rk, dims)
    after = (R.STATS[0], R.STATS_BACK[0], R.STATS_PBPK[0])
    how = ("reblock" if after[0] > before[0] else "pbpk" if after[2] > before[2]
           else "reblock_back" if after[1] > before[1] else "stock")
    mc = rk.get("memory_config")
    routes[("taped_permute", tuple(dims), _sig(ra[0]) if len(ra) else "?",
            str(mc.buffer_type) if mc is not None else "same", how)] += 1
    return res


TT._permute_fwd = permute_fwd

back0 = T._channel_move_back


def channel_move_back(chunk, memory_config):
    a = R.eligible_back(chunk, memory_config)
    b = R.eligible_back(chunk, memory_config, taped_ok=True)
    from tt_bio import ops
    # A raw operand under an open tape skips the shim: ttnn.permute is then the stock op.
    routes[("_channel_move_back", _sig(chunk), str(memory_config.buffer_type),
            "direct" if a else "taped_permute" if b else "two_transposes",
            "on_tape" if TT._on_tape((chunk,), {}) else "raw",
            "recording" if ops.recording() else "not_recording")] += 1
    return back0(chunk, memory_config)


T._channel_move_back = channel_move_back

sys.argv = [str(ROOT / "perf/spd/bc2_evo_devprof.py")] + argv
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    doc = dict(argv=argv,
               routes=[dict(key=list(k), calls=v) for k, v in routes.most_common()],
               rejects=[dict(reason=k[0], shape=list(k[1]), calls=v)
                        for k, v in sorted(R.REJECTS.items(), key=lambda kv: -kv[1])],
               stats=dict(fwd=R.STATS, back=R.STATS_BACK, pbpk=R.STATS_PBPK))
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1)[:6000])
