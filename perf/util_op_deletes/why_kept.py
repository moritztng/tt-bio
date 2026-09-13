#!/usr/bin/env python3
"""Which branch keeps the trimul's separate operand transpose, and how often.

`flag_fired.py` proved the D4 lever is live but could not say what the calls it misses do, and a
single process runs more than one geometry: `baseline_attrib.py` folds a cold 512 aa target before
any phase, so a `--phases control` run counts 512 aa calls beside the 298 aa ones. That is what
made a whole-process total read like a partial lever. This prints the census the trimul keeps,
keyed by branch, so one run says which geometry defers and which does not.

    python3 perf/util_op_deletes/why_kept.py --phases control --reps 1 --size 512 --out ... --cifdir ...
"""
import atexit, runpy, sys

import tt_bio.tenstorrent as T


@atexit.register
def _report() -> None:
    st = T.TRIMUL_MM_TRANSPOSE_STATS
    tot = sum(st.values())
    d = sum(v for (_b, w), v in st.items() if w == "defer")
    print("WHY flag=%s deferred=%d of %d" % (T._TRIMUL_MM_TRANSPOSE, d, tot),
          file=sys.stderr, flush=True)
    for (branch, what), v in sorted(st.items(), key=lambda kv: -kv[1]):
        print("WHY  %6d  %-18s %s" % (v, branch, what), file=sys.stderr, flush=True)


sys.argv[0] = "baseline_attrib.py"
runpy.run_path("perf/b2x-baseline-attrib/baseline_attrib.py", run_name="__main__")
