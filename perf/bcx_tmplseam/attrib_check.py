#!/usr/bin/env python3
"""Which op_names `parts.py` charges 145.8 s/round to, named rather than guessed.

`parts.py` and `opname.py` share one instrument -- the same trace, the same
`hostmap.exclusive_segments`, the same `wall_split`, the same `opname.op_names` mapping. They
differ in one thing, the regex. `opname.py` matches `template|structure_module` and reports
2.585 s per round; `parts.py` adds `fold_iteration|invariant_point|rigid_sidechain` and reports
145.8 s. So 143 s sits on op_names that carry a structure-module SUBMODULE name and not the
module's own, and this prints them: the distinct full op_names, their wall, their segment count
and their median segment, plus what the rest of the round is made of at the same grain.
"""
import collections
import glob
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/bcx-tmplseam/perf/bcx_seam")
sys.path.insert(0, "/home/ttuser/.coworker/wt/bcx-tmplseam/perf/bcx_tmplseam")
import hostmap as H                                                    # noqa: E402
import opname as O                                                     # noqa: E402

RUN = sys.argv[1]
SUB = re.compile(r"fold_iteration|invariant_point|rigid_sidechain")
OWN = re.compile(r"structure_module")
TMPL = re.compile(r"template")

from jax.profiler import ProfileData                                   # noqa: E402

ev = json.load(open(f"{RUN}/round_events.json"))
xp = sorted(glob.glob(f"{RUN}/trace/**/*.xplane.pb", recursive=True))
pd = ProfileData.from_file(xp[-1])
thunks = collections.defaultdict(list)
programs = collections.Counter()
for pl in pd.planes:
    for ln in pl.lines:
        for e in ln.events:
            st = dict(e.stats)
            if "hlo_op" not in st or e.name.startswith("end: "):
                continue
            programs[(st.get("hlo_module"), st.get("program_id"))] += 1
            if st.get("hlo_module") != "jit_sequence_design_loss":
                continue
            thunks[ln.name].append((e.start_ns / 1e9, e.end_ns / 1e9, st["hlo_op"]))
pid = max((p for p in programs if p[0] == "jit_sequence_design_loss"),
          key=lambda p: programs[p])[1]
hlo = sorted(glob.glob(f"{RUN}/hlo/module_{int(pid):04d}.jit_sequence_design_loss"
                       ".cpu_after_optimizations.txt"))
names = O.op_names(hlo)

tr0 = next(e["t0"] for e in ev["events"] if e["kind"] == "trace_start")
tr1 = next((e["t0"] for e in ev["events"] if e["kind"] == "trace_stop"), float("inf"))
n = sum(1 for e in ev["events"] if e["phase"] == "sequence_gradients"
        and tr0 <= e["t0"] and e["t1"] <= tr1)

segs = [s for line in thunks.values() for s in H.exclusive_segments(line)]
wall = H.wall_split(segs)

# Per distinct op_name: wall, count, and the raw (non-split) duration, which is what a
# concurrency artifact shows up in.
agg = collections.defaultdict(lambda: {"wall": 0.0, "raw": 0.0, "n": 0, "each": []})
total = 0.0
for i, (a, b, op) in enumerate(segs):
    opn = names.get(op, "<unmapped>")
    total += wall[i]
    r = agg[opn]
    r["wall"] += wall[i]
    r["raw"] += b - a
    r["n"] += 1
    if len(r["each"]) < 200000:
        r["each"].append(wall[i])

buckets = collections.Counter()
braw = collections.Counter()
bn = collections.Counter()
for opn, r in agg.items():
    if OWN.search(opn) or TMPL.search(opn):
        k = "matched by opname.py (template|structure_module)"
    elif SUB.search(opn):
        k = "ONLY by parts.py (fold_iteration|invariant_point|rigid_sidechain)"
    elif opn in ("<unlabelled>", "<unmapped>"):
        k = opn
    else:
        k = "everything else"
    buckets[k] += r["wall"]
    braw[k] += r["raw"]
    bn[k] += r["n"]

extra = [(opn, r) for opn, r in agg.items()
         if SUB.search(opn) and not OWN.search(opn) and not TMPL.search(opn)]
extra.sort(key=lambda kv: -kv[1]["wall"])

out = {
    "run": RUN,
    "profiled_rounds": n,
    "thunk_wall_s_per_round": round(total / n, 3),
    "distinct_op_names": len(agg),
    "buckets_s_per_round": {k: round(v / n, 3) for k, v in buckets.most_common()},
    "buckets_rawsum_s_per_round": {k: round(braw[k] / n, 3) for k, _ in buckets.most_common()},
    "buckets_segments": {k: bn[k] for k, _ in buckets.most_common()},
    "parts_only_top": [
        {"op_name": opn,
         "wall_s_per_round": round(r["wall"] / n, 4),
         "rawsum_s_per_round": round(r["raw"] / n, 4),
         "segments": r["n"],
         "median_seg_ms": round(1000 * statistics.median(r["each"]), 4),
         "max_seg_ms": round(1000 * max(r["each"]), 4)}
        for opn, r in extra[:25]],
}
pathlib.Path(RUN, "attrib_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
