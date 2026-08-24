#!/usr/bin/env python3
"""boltz2 at a REAL size, measured with the gate's own warm protocol.

`scripts/perf_regression.py` hardcodes trpcage (20 aa) as the fold input for every fold model, so
the 32-vs-64 answer it gives is a 20 aa answer. That is the one size where the two multiples are
NOT both compute-bound, and it is the size that produced the +4.18% reading for 64.

This reuses that module's `measure()` unchanged -- same 2 warmup + 5 timed, same median, same
device open -- and only repoints the fold input. Nothing in the gate file is modified, so the gate
keeps measuring exactly what it measured before; the constant is rebound in THIS process only.

8HEL (76 aa) is deliberate: it is the size openfold3's 32-vs-64 arm was measured at (64 costs
+0.5% there), so a boltz2 number at the same length turns a cross-MODEL comparison into a
same-length one. At 76 the two multiples genuinely differ -- 32 pads to 96, 64 pads to 128, which
is 2.37x the triangle work.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import perf_regression as PR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--input", default="examples/8hel_nomsa.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    inp = (ROOT / a.input).resolve()
    if not inp.exists():
        raise SystemExit(f"missing fold input {inp}")
    # The one rebinding. `measure()` reads the module global, so this is the whole override.
    PR.TRPCAGE = inp

    res = PR.measure(a.model, pathlib.Path(a.out))
    res["fold_input"] = str(inp)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))
    print("[%s] %.6f %s  (median %.4f s, times %s)  input=%s"
          % (a.model, res["throughput"], res["unit"], res["median_s"],
             res["times_s"], inp.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
