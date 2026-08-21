"""Run an existing harness with the round-to-nearest-even bfloat16 add installed.

    python3 scripts/bf16_add/run_with_rne.py scripts/esmfold2_e2e_parity.py --target trp_cage

Set ``RNE_ADD=0`` to run the same command with the patch NOT installed, so the A and B arms
differ only by the arithmetic and nothing else (same wrapper, same argv, same sys.path).
"""
import atexit
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rne_add  # noqa: E402

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

target = sys.argv[1]
sys.argv = sys.argv[1:]
on = os.environ.get("RNE_ADD", "1") != "0"
if on:
    rne_add.install()
atexit.register(lambda: print(f"[rne_add arm={\"on\" if on else \"off\"}] {rne_add.report()}",
                              file=sys.stderr, flush=True))
sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
runpy.run_path(target, run_name="__main__")
