"""Run the page's own RFD3 design instrument and report whether the levers fired.

`perf/dsfix/rfd3_batch_e2e.py` is the harness that produced every published `rfd3` cell
(4 designs in one `run_design`, cold chunk dropped, median of the warm chunks). It says
nothing about which optional kernels served the run, so a levers-on arm and a levers-off
arm are indistinguishable in its JSONL. This driver runs it unchanged and prints the two
lever counters at exit, so "the lever fired at this atom count" is measured rather than
assumed:

    dense  -- tt_bio.rfd3_bias.DSTATS (L2, fused dense scores+bias fp32)
    bf16   -- tt_bio.softmax_generic.SSTATS (L5a, fused bf16-packing softmax)

Usage is the harness's own argv:

    ~/.coworker/scripts/benchlock.sh <owner> -- env TT_VISIBLE_DEVICES=0 ... \
      python3 -u perf/p78/page_cell.py R4 1 +arm
"""
import atexit
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import tt_bio.rfd3_bias as rfd3_bias                # noqa: E402
import tt_bio.softmax_generic as softmax_generic    # noqa: E402


@atexit.register
def _levers():
    print("[levers] dense_bias_fused served=%d declined=%d | softmax_bf16 served=%d | %s"
          % (rfd3_bias.DSTATS[0], rfd3_bias.DSTATS[1], softmax_generic.SSTATS[0],
             rfd3_bias.stats_line()), flush=True)


runpy.run_path("perf/dsfix/rfd3_batch_e2e.py", run_name="__main__")
