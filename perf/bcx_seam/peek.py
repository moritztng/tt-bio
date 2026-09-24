#!/usr/bin/env python3
"""Top thunks of one hostmap label with their op_name: peek.py <run_dir> '<label substring>'."""
import os, sys, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostmap as H, glob, json
from jax.profiler import ProfileData
run = sys.argv[1]; want = sys.argv[2]
hlo = [json.load(open(f"{run}/hostmap.json"))["hlo"]]  # the program that ran
# op_name per instruction
opn = {}
for p in hlo[:1]:
    for line in open(p):
        m = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=", line)
        mm = H.META.search(line)
        if m and mm: opn.setdefault(m.group(1), mm.group(1))
labels = H.parse_hlo(hlo[:1]); labels.pop("__inherited__")
pd = ProfileData.from_file(sorted(glob.glob(f"{run}/trace/**/*.xplane.pb", recursive=True))[-1])
tot = collections.Counter()
for pl in pd.planes:
    for ln in pl.lines:
        segs = [(e.start_ns/1e9, e.end_ns/1e9, dict(e.stats)["hlo_op"]) for e in ln.events if "hlo_op" in dict(e.stats) and not e.name.startswith("end: ")]
        for a, b, op in H.exclusive_segments(segs):
            l = labels.get(op)
            if l and want in f"{l[0]} [{l[1]}] {l[2]}":
                tot[op] += b - a
print(hlo[0])
for op, t in tot.most_common(15):
    print(round(t, 3), op, (opn.get(op) or "(inherited)")[-160:])
