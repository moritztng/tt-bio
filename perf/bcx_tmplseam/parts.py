#!/usr/bin/env python3
"""Group the profiled round's thunks into the four parts this row has to tell apart."""
import collections
import glob
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/home/ttuser/.coworker/wt/bcx-tmplseam/perf/bcx_seam")
sys.path.insert(0, "/home/ttuser/.coworker/wt/bcx-tmplseam/perf/bcx_tmplseam")
import hostmap as H
import opname as O

RUN = sys.argv[1]

PARTS = [
    ("template: pair stack", r"template_pair_stack"),
    ("template: pointwise attention", r"template_embedding.*attention|template_pointwise"),
    ("template: single embedding", r"single_template_embedding|template_embedding"),
    ("template: single/torsion", r"template_single_embedding|template_projection"),
    ("structure module", r"structure_module|fold_iteration|invariant_point|rigid_sidechain"),
]


def part_of(opn):
    for label, pat in PARTS:
        if re.search(pat, opn):
            return label
    return None


from jax.profiler import ProfileData
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
sg = sum(e["dt"] for e in ev["events"] if e["phase"] == "sequence_gradients"
         and tr0 <= e["t0"] and e["t1"] <= tr1)
segs = [s for line in thunks.values() for s in H.exclusive_segments(line)]
wall = H.wall_split(segs)
by = collections.Counter()
total = 0.0
for i, (_a, _b, op) in enumerate(segs):
    opn = names.get(op, "<unmapped>")
    total += wall[i]
    p = part_of(opn)
    if p is None:
        continue
    by[f"{p} [{'bwd' if 'transpose(' in opn else 'fwd'}]"] += wall[i]
out = {"run": RUN, "profiled_rounds": n,
       "sequence_gradients_s_per_round": round(sg / n, 3),
       "thunk_wall_s_per_round": round(total / n, 3),
       "parts_s_per_round": {k: round(v / n, 4) for k, v in by.most_common()},
       "parts_total_s_per_round": round(sum(by.values()) / n, 3)}
pathlib.Path(RUN, "parts.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
