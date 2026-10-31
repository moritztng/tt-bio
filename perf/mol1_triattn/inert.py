#!/usr/bin/env python3
"""Do the two any-width fixes reach an existing model? Runs perf/spd/bench.py in-process with both resolvers
wrapped: every `_mm_block_at` call that the new kt >= 8 default answers (main returned None there), and every
triatt_ln `_plan` that picks a column-group count main never tried (anything but 1, 2, 4), is counted. Zero of
both means the fold ran main code paths exactly. Usage: inert.py <out.json> <bench.py args...>
"""
import json, os, runpy, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tt_bio.tenstorrent as T
import tt_bio.triatt_ln as TL

assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
hits = {"mm_default_wide": Counter(), "plan_new_group": Counter(), "mm_calls": 0, "plan_calls": 0}
_mm, _plan = T._mm_block_at, TL._plan


def mm(kt, nt):
    hits["mm_calls"] += 1
    blk = _mm(kt, nt)
    old = T._MM_BLOCK.get((kt, nt))
    old = old if old is not None else T._mm_fused_block(kt, nt)
    if blk is not None and old is None:
        hits["mm_default_wide"][f"{kt}x{nt}"] += 1
    return blk


def plan(kt, nw, tr, dev, ncores):
    hits["plan_calls"] += 1
    p = _plan(kt, nw, tr, dev, ncores)
    if p is not None and p[0] not in (1, 2, 4):
        hits["plan_new_group"][f"kt{kt}_nw{nw}_g{p[0]}"] += 1
    return p


def dump():
    out.write_text(json.dumps({k: (dict(v) if isinstance(v, Counter) else v) for k, v in hits.items()}, indent=1))


def _exit(rc, _real=os._exit):  # bench.py leaves through os._exit, which skips finally
    dump()
    _real(rc)


T._mm_block_at, TL._plan, os._exit = mm, plan, _exit
out = Path(sys.argv[1])
sys.argv = [str(ROOT / "perf/spd/bench.py")] + sys.argv[2:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    dump()
