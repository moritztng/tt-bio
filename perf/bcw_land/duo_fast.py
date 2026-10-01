#!/usr/bin/env python3
"""`duo_round.py` with every predictor built at memory="fast" instead of "auto", so the stack arm
runs the slowmode code path with the fast mode named explicitly. Prints the mode each axis ran."""
import atexit, pathlib, runpy, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt_bio import bindcraft2

_init, _seen = bindcraft2._Memory.__init__, []


def _fast(self, requested="auto"):
    _init(self, "fast")
    _seen.append(self)


bindcraft2._Memory.__init__ = _fast
atexit.register(lambda: print(f"bcw-land: memory used {[m.used for m in _seen]}", flush=True))
sys.argv[0] = str(ROOT / "perf/bcx_p10_duotraj/duo_round.py")
runpy.run_path(sys.argv[0], run_name="__main__")
