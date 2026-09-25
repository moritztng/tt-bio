#!/usr/bin/env python3
"""Group a profiled round's thunks by their FULL optimised-HLO `op_name`.

`perf/bcx_seam/hostmap.py` buckets into modules; this keeps the haiku path. Same instrument,
same exclusive-segment and wall-split treatment (it imports them), no alignment to the meter's
clock: the profiler window IS the rounds asked for, so the report divides by how many
`sequence_gradients` calls fall inside it rather than charging each round separately.
"""
import argparse
import collections
import glob
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "bcx_seam"))
import hostmap as H                                                    # noqa: E402

META = re.compile(r'op_name="([^"]*)"')


def op_names(paths):
    """instruction name -> op_name, fusions voting over their fused body, as hostmap does."""
    own, calls, body, where = {}, {}, collections.defaultdict(list), {}
    for path in paths:
        comp = None
        for line in open(path):
            m = H.COMP.match(line)
            if m and "=" not in line.split("{")[0]:
                comp = m.group(1)
                continue
            if "=" not in line:
                continue
            im = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=", line)
            if not im:
                continue
            name = im.group(1)
            mm = META.search(line)
            opn = mm.group(1) if mm else None
            if opn is not None:
                body[comp].append(opn)
            own.setdefault(name, opn)
            where.setdefault(name, comp)
            cm = re.search(r"calls=%?([\w.\-]+)", line)
            if cm:
                calls[name] = cm.group(1)
    out = {}
    for name, opn in own.items():
        voters = body.get(calls.get(name), [])
        if voters:
            out[name] = collections.Counter(voters).most_common(1)[0][0]
        elif opn:
            out[name] = opn
        else:
            # NOT the enclosing computation's majority. `hostmap.parse_hlo` can afford that
            # because it votes over coarse module LABELS; voting over a raw `op_name` charges
            # every unlabelled instruction in the entry computation to whatever module happens
            # to be commonest in it, which on this run put 163 s of a 181 s round on the
            # structure module, a hundred times what `hostmap.py` charges the same module. An
            # instruction with no `op_name` of its own stays unlabelled and is not counted.
            out[name] = "<unlabelled>"
    return out


def shorten(opn):
    """The haiku path without the jvp/transpose wrappers, plus the direction."""
    bwd = "transpose(" in opn
    path = opn.split("/")
    keep = [p for p in path if p and not p.startswith(("jit(", "jvp(", "transpose(", "custom_"))]
    return ("bwd" if bwd else "fwd"), "/".join(keep[-6:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--match", default="template|structure_module",
                    help="regex over the op_name; only matching thunks are broken out")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    from jax.profiler import ProfileData
    ev = json.load(open(f"{args.run_dir}/round_events.json"))
    xp = sorted(glob.glob(f"{args.run_dir}/trace/**/*.xplane.pb", recursive=True))
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
    hlo = sorted(glob.glob(f"{args.run_dir}/hlo/module_{int(pid):04d}."
                           "jit_sequence_design_loss.cpu_after_optimizations.txt"))
    assert hlo, f"no dump for program_id {pid}"
    names = op_names(hlo)

    tr0 = next(e["t0"] for e in ev["events"] if e["kind"] == "trace_start")
    tr1 = next((e["t0"] for e in ev["events"] if e["kind"] == "trace_stop"), float("inf"))
    n_rounds = sum(1 for e in ev["events"] if e["phase"] == "sequence_gradients"
                   and tr0 <= e["t0"] and e["t1"] <= tr1)
    sg_s = sum(e["dt"] for e in ev["events"] if e["phase"] == "sequence_gradients"
               and tr0 <= e["t0"] and e["t1"] <= tr1)
    segs = [s for line in thunks.values() for s in H.exclusive_segments(line)]
    wall = H.wall_split(segs)

    pat = re.compile(args.match)
    per_op = collections.Counter()
    per_group = collections.Counter()
    total = 0.0
    for i, (_a, _b, op) in enumerate(segs):
        opn = names.get(op, "<unmapped>")
        total += wall[i]
        if not pat.search(opn):
            continue
        d, short = shorten(opn)
        per_op[f"[{d}] {short}"] += wall[i]
        per_group[f"[{d}] {short.split('/')[0]}"] += wall[i]
    n = max(n_rounds, 1)
    out = {"run": args.run_dir, "hlo": hlo[0], "profiled_rounds": n_rounds,
           "sequence_gradients_s_in_window": round(sg_s, 3),
           "thunk_wall_s_in_window": round(total, 3),
           "match": args.match,
           "matched_wall_s_per_round": round(sum(per_op.values()) / n, 3),
           "per_round": {k: round(v / n, 4) for k, v in per_op.most_common(args.top)},
           "per_round_by_top_scope": {k: round(v / n, 4) for k, v in per_group.most_common(20)}}
    pathlib.Path(args.run_dir, "opname.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
