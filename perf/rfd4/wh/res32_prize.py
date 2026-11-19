"""What the T2_RES32 fused residual is worth, by taking it away on the arch that has it.

`tt_bio/trimul_tail.py` declines the float32-stream T2 form on Wormhole (`T2_RES32_ARCHS`), where it
computes a wrong update, so a Wormhole trunk pays an unfused residual add over the whole pair tensor
at every pair update. Blackhole takes the fused form. Running Blackhole with `T2_RES32_ARCHS` emptied
is Wormhole's situation on a card we can measure freely, so the gap between the two arms prices the
Wormhole-only loss before anyone tries to root-cause the kernel.

    T2_RES32_ARCHS=none python perf/rfd4/wh/res32_prize.py --weights ... --n 640 --real 620 --batch 1 --out x.json

Takes pass_bench's arguments; `T2_RES32_ARCHS=none` empties the table before the trunk is built.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio import trimul_tail

if os.environ.get("T2_RES32_ARCHS") == "none":
    trimul_tail.T2_RES32_ARCHS = ()
elif os.environ.get("T2_RES32_ARCHS"):
    trimul_tail.T2_RES32_ARCHS = tuple(os.environ["T2_RES32_ARCHS"].split(","))

runpy.run_path(str(Path(__file__).resolve().parents[1] / "e2e" / "pass_bench.py"), run_name="__main__")
