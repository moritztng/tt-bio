"""In-process BoltzGen runner for the H200 sweep, patched to stamp every diffusion step.

Upstream wraps the denoising loop in `optionally_tqdm` (diffusion.py:567). Replacing that with a
stamping generator gives the per-step wall directly -- the same measurement the TT ladder takes
from the `diff k/N` log lines, so the two sides are measured the same way rather than one by
per-step wall and the other by a step-count differential.

`--no_subprocess` is mandatory: the pipeline otherwise forks the design step and the patch would
not apply in the child.

Prints one `STEP <unix_time>` line per step to stdout; the driver reduces them.
"""
import sys, time

import boltzgen.model.modules.diffusion as D

_orig = D.optionally_tqdm


def stamping(iterable, use_tqdm=True, desc=None, **kw):
    for item in iterable:
        sys.stdout.write("STEP %.6f\n" % time.time())
        sys.stdout.flush()
        yield item


D.optionally_tqdm = stamping

from boltzgen.cli.boltzgen import main

sys.argv = ["boltzgen"] + sys.argv[1:]
main()
