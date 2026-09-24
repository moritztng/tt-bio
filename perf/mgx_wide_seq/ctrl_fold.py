"""The size ladder's own fold (release_gate._run_census_fold) with this row's two levers off.

    TT_VISIBLE_DEVICES=<c> ... python perf/mgx_wide_seq/ctrl_fold.py <model> <rung>[,<rung>...]

Its CIF is compared byte for byte with the speed ladder's fold of the same rung on the same commit
(perf/mgx_wide_seq/speed/work-<model>/out_<model>-<rung>-rep0).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.update(TT_BIO_SIZE_LIMIT="0", TT_BIO_PAIR_INPLACE="0", TT_BIO_TRIMUL_INPROJ_ROWBLOCK_NORM="0")
import release_gate as rg  # noqa: E402

rg.HOST_THREADS = 2
model = sys.argv[1]
for rung in map(int, sys.argv[2].split(",")):
    r = rg._run_census_fold(model, rung, ROOT / "perf" / "mgx_wide_seq" / "ctrl", "flagsoff")
    print(json.dumps({"model": model, "rung": rung,
                      **{k: r.get(k) for k in ("error", "refused", "runtime_s", "aiclk", "load")}},
                     default=str), flush=True)
