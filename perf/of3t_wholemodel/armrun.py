#!/usr/bin/env python3
"""Run another row's instrument under this row's arm flags, and PROVE the flags fired.

`of3t-direct`'s three scope instruments take no arm arguments -- the two repairs are selected by
environment, `TT_BIO_SOFTMAX_BW_RENORM` at import time and `TT_BIO_HOST_F64_SOFTMAX_AB` per
construction site. An environment variable that is read at import and never reported is the
shape of `a-lever-can-fire-and-be-inert`: the run succeeds, the number is the shipped number,
and nothing says which. So this wrapper runs the instrument unchanged and then reads the two
counters out of the loaded modules, and a zero served count on an arm that asked for the host
path is a hard failure here exactly as it is in `device_gradient.py`.

    armrun.py <instrument.py> [args...]
"""
from __future__ import annotations

import json
import os
import runpy
import sys

script, argv = sys.argv[1], sys.argv[2:]
want_f64 = os.environ.get("TT_BIO_HOST_F64_SOFTMAX_AB", "") not in ("", "-all")
want_renorm = os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "0").lower() in ("1", "true", "yes", "on")
print(f"ARM: renorm={want_renorm} host_f64_softmax={want_f64!r} "
      f"({os.environ.get('TT_BIO_HOST_F64_SOFTMAX_AB','')!r})", flush=True)

sys.argv = [script] + argv
rc = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    rc = e.code or 0

stats = {}
tt = sys.modules.get("tt_bio.tenstorrent")
tp = sys.modules.get("tt_bio.taped_ttnn")
ag = sys.modules.get("tt_bio.autograd")
if ag is not None:
    # The third repair arm, on the same terms as the two above: a flag that is set but never
    # reached is not an arm. calls separates "declined" from "never called" -- the diffusion
    # capture reads flag true, calls 0, and no arm of it can move.
    stats["SOFTMAX_BW_FP32"] = {"flag": bool(ag.SOFTMAX_BW_FP32),
                                **dict(ag.SOFTMAX_BW_FP32_STATS)}
if tt is not None:
    stats["HOST_F64_SOFTMAX_STATS"] = dict(tt.HOST_F64_SOFTMAX_STATS)
if tp is not None:
    stats["_SOFTMAX_BW_RENORM"] = bool(tp._SOFTMAX_BW_RENORM)
    stats["softmax_bw_calls"] = getattr(tp, "_SOFTMAX_BW_CALLS", None)
print("ARM_EVIDENCE " + json.dumps(stats), flush=True)

if want_renorm and stats.get("_SOFTMAX_BW_RENORM") is not True:
    print("FAILED: the renorm arm did not reach taped_ttnn, so this is the shipped arm under "
          "another name", flush=True)
    rc = rc or 3
want_fp32bw = os.environ.get("TT_BIO_SOFTMAX_BW_FP32", "0").lower() in ("1", "true", "yes", "on")
if want_fp32bw and stats.get("SOFTMAX_BW_FP32", {}).get("fired", 0) == 0:
    print("FAILED: TT_BIO_SOFTMAX_BW_FP32 fired 0 times, so this is the shipped arm under "
          "another name", flush=True)
    rc = rc or 3
if want_f64 and stats.get("HOST_F64_SOFTMAX_STATS", {}).get("served", 0) == 0:
    print("FAILED: the host float64 softmax served 0 calls, so this is the shipped arm under "
          "another name", flush=True)
    rc = rc or 3
sys.exit(rc)
