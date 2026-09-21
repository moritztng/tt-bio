#!/usr/bin/env python3
"""What the repair costs: how many times a training step READS `Tensor.value`.

`Tensor.value` became a property so the parameter registration follows the handle the
optimizer writes. A property get is 13.4 ns against a slot's 6.4 ns on this host
(`perf/of3t_rebind/propbench.py`), so the cost of the repair is 7.0 ns times this count.
Counted on the real thing: one step of `of3t-modeltraj`'s conditioning loop, taped forward,
backward, optimizer step and rebind.

Only the tape builds `Tensor`s, so inference reads this property zero times.
"""
import os
import runpy
import sys
import time

# The CHECKOUT, not the installed package. `python3 perf/of3t_rebind/valuecount.py` puts the
# SCRIPT s directory on sys.path[0], so a bare `import tt_bio` resolves to whatever is installed
# and this instrument would measure a tree that is not the one under test.
sys.path.insert(0, os.getcwd())

sys.argv = ["rebind_traj.py", "--arm", "shipped", "--steps", "1",
            "--out", "/tmp/of3t/rebind/valuecount_traj.json"]

import tt_bio.autograd as ag                                                # noqa: E402

N = [0]
_fset = ag.Tensor.value.fset


def _get(self):
    N[0] += 1
    return self._value


ag.Tensor.value = property(_get, _fset)

t0 = time.perf_counter()
try:
    runpy.run_path("perf/of3t_rebind/rebind_traj.py", run_name="__main__")
except SystemExit:
    pass
print(f"VALUE_READS_PER_STEP={N[0]} wall_s={time.perf_counter()-t0:.1f}", flush=True)
