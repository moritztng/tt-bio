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

# A part is a set of haiku module names matched against the op_name's PATH COMPONENTS, never
# as a substring of the whole path. The substring form was wrong by a hundredfold: `fold_iteration`
# is a substring of `alphafold_iteration`, which prefixes every op in the model, so the structure
# module's bucket silently became the whole of AlphaFold -- 143.4 s of a 166.9 s round over
# 375,096 segments, whose largest contributors are Evoformer triangle-attention dot_generals
# (perf/bcx_tmplseam/attrib_check.py, runs/anatomy/attrib_check.json).
STRUCTURE = {"structure_module", "fold_iteration", "invariant_point_attention",
             "multi_rigid_sidechain", "rigid_sidechain"}
WRAPPED = re.compile(r"^[\w.]+\((.*)\)$")


def components(opn):
    """Path components, with one level of `jit(...)`/`vmap(...)`/`transpose(...)` unwrapped."""
    out = set()
    for c in opn.split("/"):
        if not c:
            continue
        out.add(c)
        m = WRAPPED.match(c)
        while m:
            c = m.group(1)
            out.add(c)
            m = WRAPPED.match(c)
    return out


def part_of(opn):
    s = components(opn)
    if "template_pair_stack" in s:
        return "template: pair stack"
    if "template_embedding" in s and (s & {"attention", "template_pointwise_attention"}):
        return "template: pointwise attention"
    if s & {"single_template_embedding", "template_embedding"}:
        return "template: single embedding"
    if s & {"template_single_embedding", "template_projection"}:
        return "template: single/torsion"
    if s & STRUCTURE:
        return "structure module"
    return None


# The regression this file is the site of, asserted rather than remembered.
_EVO = ("jit(sequence_design_loss)/jvp(jit(apply))/jit(apply_fn)/alphafold/alphafold_iteration/"
        "evoformer/__layer_stack_no_state_1/while/body/eval_jaxpr/evoformer_iteration/"
        "triangle_attention_starting_node/attention/bqhc,bkhc->bhqk/dot_general")
assert part_of(_EVO) is None, "an Evoformer op must not land in a template or structure bucket"
assert part_of(_EVO.replace("evoformer_iteration", "structure_module")) == "structure module"


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
    p = part_of(opn) or ("<unlabelled>" if opn in ("<unlabelled>", "<unmapped>")
                         else "everything else")
    by[f"{p} [{'bwd' if 'transpose(' in opn else 'fwd'}]"] += wall[i]
out = {"run": RUN, "profiled_rounds": n,
       "sequence_gradients_s_per_round": round(sg / n, 3),
       "thunk_wall_s_per_round": round(total / n, 3),
       "parts_s_per_round": {k: round(v / n, 4) for k, v in by.most_common()},
       "parts_total_s_per_round": round(sum(by.values()) / n, 3)}
pathlib.Path(RUN, "parts.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
