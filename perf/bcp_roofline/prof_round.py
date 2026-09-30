#!/usr/bin/env python3
"""The shipped round under the device profiler, with a signpost at every round boundary.

`perf/bcx_p10_duotraj/duo_round.py` unchanged, run under `python -m tracy` on a Tracy build of the
same ttnn version the wheel ships (0.68.0). The only addition is a `tracy.signpost` each time a
round starts, so the ops report can be cut into rounds and a warm round's device kernel time is
read off the card's own timestamps instead of a host clock around an asynchronous enqueue.

Wall under the profiler is NOT the round: every program pays marker writes and the host drains
the profiler DRAM buffer. Only per-op DEVICE KERNEL DURATION and the op mix are read from here.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "perf" / "bcx_p10_duotraj"))
import duo_round as D                                                   # noqa: E402
from tracy import signpost                                              # noqa: E402


def _marked(cls):
    enter = cls.on_sequence_gradients_enter

    def mark(self, *a, **kw):
        signpost("bcp_round")
        return enter(self, *a, **kw)
    cls.on_sequence_gradients_enter = mark


_marked(D.M.Meter)
if "on_sequence_gradients_enter" in D.DuoMeter.__dict__:
    _marked(D.DuoMeter)


def _seams():
    """A begin/end signpost around every device seam, so each op lands in (module, phase)."""
    from tt_bio import bindcraft2 as B
    for module, cls in (("evoformer", B.EvoformerOnDevice), ("extra_msa", B.ExtraMsaOnDevice),
                        ("template", B.TemplateOnDevice)):
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name)

            def make(orig, tag):
                def wrapper(self, *a, **kw):
                    signpost(f"seam_begin:{tag}")
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        signpost(f"seam_end:{tag}")
                return wrapper
            setattr(cls, name, make(orig, f"{module}:{name.lstrip('_')}"))


_seams()

if __name__ == "__main__":
    D.main()
