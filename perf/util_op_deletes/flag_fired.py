#!/usr/bin/env python3
"""Did TT_BIO_TRIMUL_MM_TRANSPOSE actually fire inside a real fold?

The D4 A/B reports one CIF digest across both arms, which is what a correct layout change should
do -- and is also exactly what a DEAD flag would do. So run the fold under the real driver and
read the census the trimul keeps per matmul call. Arm 0 must be all `keep`, arm 1 must be all
`defer`; anything else means the number the A/B measured is not the number this lever is supposed
to move. For the per-branch breakdown use `why_kept.py`.
"""
import atexit, runpy, sys

import tt_bio.tenstorrent as T


@atexit.register
def _report() -> None:
    st = T.TRIMUL_MM_TRANSPOSE_STATS
    deferred = sum(v for (_b, w), v in st.items() if w == "defer")
    kept = sum(v for (_b, w), v in st.items() if w == "keep")
    print("FIRED flag=%s deferred=%d kept=%d" % (
        T._TRIMUL_MM_TRANSPOSE, deferred, kept), file=sys.stderr, flush=True)


sys.argv[0] = "baseline_attrib.py"
runpy.run_path("perf/b2x-baseline-attrib/baseline_attrib.py", run_name="__main__")
