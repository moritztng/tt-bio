"""Run the page's own RFD3 design instrument and report whether the levers fired.

`perf/dsfix/rfd3_batch_e2e.py` is the harness that produced every published `rfd3` cell
(4 designs in one `run_design`, cold chunk dropped, median of the warm chunks). It says
nothing about which optional kernels served the run, so a levers-on arm and a levers-off
arm are indistinguishable in its JSONL. This driver runs it unchanged and prints the two
lever counters at exit, so "the lever fired at this atom count" is measured rather than
assumed:

    dense  -- tt_bio.rfd3_bias.DSTATS (L2, fused dense scores+bias fp32)
    bf16   -- tt_bio.softmax_generic.SSTATS (L5a, fused bf16-packing softmax)

It also prices co-tenant noise per chunk. benchlock gates the box at the moment it hands
over the lock and never again, so a run can start at loadavg 0.00 and finish at 15.57 --
which is exactly what runs 1 and 2 of this task did, and why their warm spread was 9.94 %
and 6.84 % against the 0.52 % the published cell holds. Per chunk this records the CPU
seconds burned by everything that is NOT this process (host busy jiffies from /proc/stat
minus our own utime+stime), so a chunk contaminated by a foreign build is identifiable
after the fact instead of being silently folded into the median.

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
from tt_bio.rfd3.sampler import RFD3Sampler         # noqa: E402

HZ = os.sysconf("SC_CLK_TCK")
NCPU = os.cpu_count() or 1
CHUNKS: list[dict] = []


def _host_busy_s():
    """Host-wide CPU seconds spent outside idle+iowait, summed over every core."""
    f = open("/proc/stat").readline().split()
    v = [int(x) for x in f[1:]]
    return (sum(v) - v[3] - v[4]) / HZ


def _self_cpu_s():
    v = os.times()
    return v[0] + v[1] + v[2] + v[3]


# The harness rebinds RFD3Sampler.sample at import time, so wrapping here first makes this
# the inner call: one entry per timed chunk, aligned with the harness's own WALLS list.
_inner = RFD3Sampler.sample


def _metered(self, dm, n, *a, **kw):
    h0, s0 = _host_busy_s(), _self_cpu_s()
    try:
        return _inner(self, dm, n, *a, **kw)
    finally:
        foreign = (_host_busy_s() - h0) - (_self_cpu_s() - s0)
        CHUNKS.append({"foreign_cpu_s": round(foreign, 1),
                       "load1": float(open("/proc/loadavg").read().split()[0])})


RFD3Sampler.sample = _metered


@atexit.register
def _report():
    print("[levers] dense_bias_fused served=%d declined=%d | softmax_bf16 served=%d | %s"
          % (rfd3_bias.DSTATS[0], rfd3_bias.DSTATS[1], softmax_generic.SSTATS[0],
             rfd3_bias.stats_line()), flush=True)
    print("[cotenant] ncpu=%d per-chunk foreign_cpu_s/load1: %s"
          % (NCPU, " ".join("%.0f/%.2f" % (c["foreign_cpu_s"], c["load1"]) for c in CHUNKS)),
          flush=True)


runpy.run_path("perf/dsfix/rfd3_batch_e2e.py", run_name="__main__")
