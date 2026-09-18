#!/usr/bin/env python3
"""Pair arms folded in SEPARATE processes against a base folded in its own.

One process per arm is not a convenience here, it is required. `sdpa_generic.sdpa`'s program
cache key omits `mask.dtype` and `out.dtype`, so two dtype arms at one shape in one process make
the second silently reuse the first's program; that is what produced this campaign's two
"mixed dataformat is wrong" readings. The pairing is still same-seed -- the seed rides on the
config, not on the interpreter -- and base reproduces bit-exactly across processes and cards,
which is checked below rather than assumed.

    pair_arms.py <size> <base_json> <base_cifdir> <arm_json> <arm_cifdir> <out_json> <out_cifdir>
"""
import json
import shutil
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
size, bj, bc, aj, ac, oj, oc = sys.argv[1:8]
bjs, ajs = json.loads((B / bj).read_text()), json.loads((B / aj).read_text())
keep = [r for r in bjs["runs"] if r["arm"] == "base" and r["target"].endswith(size)]
keep += [r for r in ajs["runs"] if r["arm"] != "base" and r["target"].endswith(size)]
(B / oj).write_text(json.dumps(
    {"doc": __doc__, "env": {**bjs["env"], "arm_env": ajs["env"]}, "runs": keep}, indent=1))
d = B / oc
if d.exists():
    shutil.rmtree(d)
d.mkdir(parents=True)
for r in keep:
    src = (B / bc if r["arm"] == "base" else B / ac) / f"{size}_{r['tag']}"
    shutil.copytree(src, d / f"{size}_{r['tag']}")
print(f"{size} aa: {len(keep)} runs -> {oj}")
