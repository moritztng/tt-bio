#!/usr/bin/env python3
"""Run another row's trunk instrument under this row's mask arm, and PROVE the arm fired.

`of3t-modelboundary`'s `runarm.sh` drives `perf/of3t_bwdaccum/dev_cot.py`, which takes no mask
argument: D174's lever is selected by `TT_BIO_MASK_TRANS`, read once at `tt_bio.tenstorrent`
import time. A flag read at import and never reported is the shape of
`a-lever-can-fire-and-be-inert`, and AMENDMENT 2 asks for the mask state recorded explicitly
because a null there is what cost this campaign several passes twice. So this wrapper runs the
instrument unchanged and then reads the counters back out of the LOADED module.

A masked arm that reports zero stacks is a hard failure here, not a null result.

    maskrun.py <sidecar.json> <instrument.py> [args...]
"""
from __future__ import annotations

import json
import os
import runpy
import sys

sidecar, script, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
asked = os.environ.get("TT_BIO_MASK_TRANS", "")
asked_ones = os.environ.get("TT_BIO_MASK_TRANS_ONES", "")
print(f"ARM: TT_BIO_MASK_TRANS={asked!r} TT_BIO_MASK_TRANS_ONES={asked_ones!r} "
      f"TT_BIO_SOFTMAX_BW_RENORM={os.environ.get('TT_BIO_SOFTMAX_BW_RENORM','')!r}", flush=True)

sys.argv = [script] + argv
rc = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    rc = e.code or 0

ev = {"mask_trans_asked": asked, "mask_trans_ones_asked": asked_ones}
T = sys.modules.get("tt_bio.tenstorrent")
if T is not None:
    ev["mask_trans_live"] = bool(T._MASK_TRANS)
    ev["mask_trans_ones_live"] = bool(T._MASK_TRANS_ONES)
    ev["MASK_TRANS_STATS"] = dict(T.MASK_TRANS_STATS)
else:
    ev["mask_trans_live"] = None
tp = sys.modules.get("tt_bio.taped_ttnn")
if tp is not None:
    ev["softmax_bw_renorm_live"] = bool(tp._SOFTMAX_BW_RENORM)
print("ARM_EVIDENCE " + json.dumps(ev), flush=True)
with open(sidecar, "w") as f:
    json.dump(ev, f, indent=1)

want = asked.strip().casefold() in ("1", "true", "yes", "on")
if want and ev.get("MASK_TRANS_STATS", {}).get("stacks", 0) == 0:
    print("FAILED: the mask arm reached no Pairformer stack, so this is the unmasked arm "
          "under another name", flush=True)
    rc = rc or 3
if not want and ev.get("MASK_TRANS_STATS", {}).get("stacks", 0) != 0:
    print("FAILED: the control arm took the masked branch", flush=True)
    rc = rc or 3
sys.exit(rc)
