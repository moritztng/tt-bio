#!/usr/bin/env python3
"""Pair a solo-process arm against base folds taken in another process.

`TT_BIO_TRIATT_BIAS_B8` cannot be measured in the same process as `base`: the SDPA program cache
key omits `mask.dtype`, so the second arm silently reuses the first arm's program. So its folds
run alone, and the pairing is done here instead. This is still a same-seed paired comparison --
the seed is on the config, not on the interpreter, and base reproduces bit-exactly across
processes and cards (verified: every base digest below matches the interleaved run).
"""
import json
import shutil
import sys
from pathlib import Path

B = Path("/home/ttuser/.coworker/wt/bfp8-accuracy-envelope/perf/bfp8_envelope")


def merge(size, base_json, base_cif, arm_json, arm_cif, out_json, out_cif):
    bj = json.loads((B / base_json).read_text())
    aj = json.loads((B / arm_json).read_text())
    keep = [r for r in bj["runs"] if r["arm"] == "base" and r["target"].endswith(size)]
    keep += [r for r in aj["runs"] if r["arm"] != "base" and r["target"].endswith(size)]
    out = {"doc": __doc__, "env": {**bj["env"], "arm_env": aj["env"]}, "runs": keep}
    (B / out_json).write_text(json.dumps(out, indent=1))
    d = B / out_cif
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for r in keep:
        src = (B / base_cif if r["arm"] == "base" else B / arm_cif) / f"{size}_{r['tag']}"
        shutil.copytree(src, d / f"{size}_{r['tag']}")
    print(f"{size} aa: {len(keep)} runs -> {out_json}")


for size in ("298", "512"):
    merge(size, f"acc_{size}.json", f"cif_{size}", "solo_Tbias_full.json", "cif_solo_Tbias_full",
          f"pair_Tbias_{size}.json", f"cif_pair_Tbias_{size}")
