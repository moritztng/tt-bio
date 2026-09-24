"""One timed size-ladder fold per rung on this commit, levers as the environment sets them.

    TT_VISIBLE_DEVICES=<c> ... python perf/mgx_wide_seq/fold1.py <model> <rung>[,<rung>...] <tag>

The speed ladder's own fold (release_gate._run_census_fold, host threads capped at 2 as in
time_rungs.py) without its warm-up: for a commit whose change since the ladder only moves
bytes faster, one fold beside the ladder's rep shows the time and, byte for byte, the output.
Folds land in perf/mgx_wide_seq/ctrl/out_<model>-<rung>-<tag>/.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["TT_BIO_SIZE_LIMIT"] = "0"
import release_gate as rg  # noqa: E402

from tt_bio import runtime  # noqa: E402

os.environ.update(runtime.host_thread_cap_env(1, 2))
model, rungs, tag = sys.argv[1], sys.argv[2], sys.argv[3]
for rung in map(int, rungs.split(",")):
    r = rg._run_census_fold(model, rung, ROOT / "perf" / "mgx_wide_seq" / "ctrl", tag)
    print(json.dumps({"model": model, "rung": rung, "tag": tag,
                      **{k: r.get(k) for k in ("error", "refused", "runtime_s", "aiclk", "load")}},
                     default=str), flush=True)
