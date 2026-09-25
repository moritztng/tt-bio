#!/usr/bin/env python3
"""Attribute a BindCraft 2 gradient round's host seconds to AF2 modules.

INSTRUMENT. `jax.profiler` on XLA CPU records one event per executed thunk on the thread
that ran it, named by its optimised-HLO instruction (`hlo_op`). The optimised HLO dump
(`--xla_dump_to`) carries each instruction's `op_name`, which is the haiku/JAX name scope it
was traced under, so a thunk maps to a module path and to a direction (`transpose(` in the
path is the backward). The device calls are `pure_callback` custom-calls and show up as
thunks too, which is what lines the profile up with `perf/bcx_round/meter.py`'s clock.

WHAT IT LIES ABOUT, and what this does about each:
* Nested thunks (a `while` around a `layer_stack` scan) contain their children on the same
  thread. Only EXCLUSIVE time is counted, so a scan is not charged twice.
* Concurrent thunks on different threads overlap. WALL seconds split every instant equally
  among the thunks running in it; CPU seconds (the plain sum) are reported beside them.
  Eigen pool workers carry no thunk name, so a matmul's helper threads are charged to the
  thunk that forked them, which is correct for wall and undercounts CPU.
* Time outside every thunk inside `sequence_gradients` is Python and JAX dispatch, not XLA.
  It is its own line ("outside XLA"), with BindCraft 2's per-call `lower().compile()` split
  out of it from the meter's own timestamp.
* The profiler itself costs time; the profiled round's wall is reported next to the
  unprofiled rounds of the same run so the overhead is visible, not assumed.
* A fusion carries one op_name while fusing ops from neighbouring scopes; it is charged to
  the most common module among its fused instructions.
"""
import collections
import glob
import json
import re
import sys

MODULES = [  # first match wins; order matters (a head inside structure_module etc.)
    ("device: trunk callback", r"(pure_callback|python_cpu_callback|xla_ffi_python)"),
    ("extra-MSA stack", r"extra_msa_stack"),
    ("template embedding", r"template_embedding|template_pair|single_template|template_pointwise"),
    ("template single/torsion", r"template_single_embedding|template_projection"),
    ("structure module", r"structure_module"),
    ("heads: pLDDT", r"predicted_lddt_head"),
    ("heads: PAE/pTM", r"predicted_aligned_error_head"),
    ("heads: distogram", r"distogram_head"),
    ("heads: masked MSA", r"masked_msa_head"),
    ("heads: exp. resolved", r"experimentally_resolved_head"),
    ("single_activations", r"single_activations"),
    ("extra-MSA embedding", r"extra_msa_activations"),
    ("embedder: recycling", r"prev_pos_linear|prev_msa_first_row_norm|prev_pair_norm"),
    ("embedder: relpos", r"pair_activiations|position_activations|relpos"),
    ("embedder: MSA/target", r"preprocess_1d|preprocess_msa|left_single|right_single"),
    ("evoformer (JAX side)", r"evoformer"),
    ("alphafold: other", r"alphafold"),
]

INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*.*?(?:calls=%?([\w.\-]+))?.*$")
META = re.compile(r'op_name="([^"]*)"(?:[^}]*?source_file="([^"]*)")?(?:[^}]*?source_line=(\d+))?')
COMP = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")


def module_of(op_name, src):
    for label, pat in MODULES:
        if re.search(pat, op_name):
            return label
    s = src or ""
    if "losses" in s or "loss" in op_name:
        return "loss"
    if s:
        base = s.rsplit("/", 1)[-1]
        return f"outside alphafold: {base}"
    return "outside alphafold: unlabelled"


SUBS = ("triangle_multiplication_outgoing", "triangle_multiplication_incoming",
        "triangle_attention_starting_node", "triangle_attention_ending_node", "pair_transition",
        "outer_product_mean", "msa_row_attention_with_pair_bias", "msa_column_global_attention",
        "msa_column_attention", "msa_transition", "template_pointwise_attention",
        "template_pair_stack", "invariant_point_attention", "rigid_sidechain", "transition")


def sub_of(opn):
    """The AF2 op inside a module, plus whether it is a remat recompute."""
    parts = [p for p in SUBS if p in opn]
    sub = parts[-1] if parts else "other"
    if "template_pair_stack" in opn and sub != "template_pair_stack":
        sub = "template_pair_stack/" + sub
    if "transpose(" not in opn and ("checkpoint" in opn or "remat" in opn):
        sub += " (remat)"
    return sub


def parse_hlo(paths):
    """instruction name -> (module, direction). Fusions vote over their fused body."""
    own, calls, body, where = {}, {}, collections.defaultdict(list), {}
    for path in paths:
        comp = None
        for line in open(path):
            m = COMP.match(line)
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
            lab = None
            if mm:
                opn = mm.group(1)
                lab = (module_of(opn, mm.group(2)),
                       "bwd" if "transpose(" in opn else "fwd", sub_of(opn))
                body[comp].append(lab)
            own.setdefault(name, lab)
            where.setdefault(name, comp)
            cm = re.search(r"calls=%?([\w.\-]+)", line)
            if cm:
                calls[name] = cm.group(1)
    # A computation's label is the majority of its labelled instructions; one with none
    # takes its caller's. An instruction XLA made up (a copy, a layout transpose) carries no
    # op_name and is charged to the computation it sits in -- a copy inside the extra-MSA
    # scan body is extra-MSA work. What still has no label is reported as unmapped.
    caller = {c: n for n, c in calls.items()}
    comp_lab = {}

    def lab_of_comp(c, depth=0):
        if c in comp_lab:
            return comp_lab[c]
        v = body.get(c)
        got = collections.Counter(v).most_common(1)[0][0] if v else None
        if got is None and depth < 8 and c in caller:
            got = lab_of_comp(where.get(caller[c]), depth + 1)
        comp_lab[c] = got
        return got
    out, inherited = {}, set()
    for name, lab in own.items():
        voters = body.get(calls.get(name), [])
        if voters:
            out[name] = collections.Counter(voters).most_common(1)[0][0]
        elif lab:
            out[name] = lab
        else:
            got = lab_of_comp(where.get(name))
            if got:
                out[name] = got
                inherited.add(name)
    out["__inherited__"] = inherited
    return out


def exclusive_segments(events):
    """Leaf-time segments on one thread: [(t0, t1, hlo_op)], nested time removed."""
    evs = sorted(events, key=lambda e: (e[0], -e[1]))
    segs, stack = [], []   # stack of [t0, t1, op, cursor]

    def emit_until(t):
        while stack and stack[-1][1] <= t:
            top = stack.pop()
            if top[3] < top[1]:
                segs.append((top[3], top[1], top[2]))
            if stack:
                stack[-1][3] = max(stack[-1][3], top[1])
    for t0, t1, op in evs:
        emit_until(t0)
        if stack and stack[-1][3] < t0:
            segs.append((stack[-1][3], t0, stack[-1][2]))
        if stack:
            stack[-1][3] = t1
        stack.append([t0, t1, op, t0])
    emit_until(float("inf"))
    return segs


def wall_split(segs):
    """Split every instant equally among the segments live in it."""
    pts = []
    for i, (a, b, _) in enumerate(segs):
        pts += [(a, 1, i), (b, -1, i)]
    pts.sort()
    live, out, last = set(), collections.Counter(), None
    for t, d, i in pts:
        if live and last is not None and t > last:
            share = (t - last) / len(live)
            for j in live:
                out[j] += share
        last = t
        (live.add if d > 0 else live.discard)(i)
    return out


def main(run_dir, rounds=None):
    from jax.profiler import ProfileData
    ev = json.load(open(f"{run_dir}/round_events.json"))
    xp = sorted(glob.glob(f"{run_dir}/trace/**/*.xplane.pb", recursive=True))
    pd = ProfileData.from_file(xp[-1])
    thunks = collections.defaultdict(list)   # thread -> [(t0, t1, op)]
    programs = collections.Counter()
    for pl in pd.planes:
        for ln in pl.lines:
            for e in ln.events:
                st = dict(e.stats)
                if "hlo_op" not in st or e.name.startswith("end: "):
                    continue
                if st.get("hlo_module") != "jit_sequence_design_loss":
                    programs[(st.get("hlo_module"), st.get("program_id"))] += 1
                    continue
                programs[("jit_sequence_design_loss", st.get("program_id"))] += 1
                thunks[ln.name].append((e.start_ns / 1e9, e.end_ns / 1e9, st["hlo_op"]))
    # The dump holds every compile of the program; the one that RAN is named by program_id,
    # which is the module's unique id in the dump's file name.
    pid = max((p for p in programs if p[0] == "jit_sequence_design_loss"),
              key=lambda p: programs[p])[1]
    hlo = sorted(glob.glob(f"{run_dir}/hlo/module_{int(pid):04d}.jit_sequence_design_loss"
                           ".cpu_after_optimizations.txt"))
    assert hlo, f"no dump for program_id {pid}"
    labels = parse_hlo(hlo)
    inherited = labels.pop("__inherited__")
    segs = [s for line in thunks.values() for s in exclusive_segments(line)]

    # Align profiler time (relative) to the meter's epoch clock on the device callbacks.
    dev = sorted((e for e in ev["events"] if e["kind"] == "device"), key=lambda e: e["t0"])
    cb = sorted((s for s in segs if labels.get(s[2], ("",))[0].startswith("device")),
                key=lambda s: s[0])
    cb_first = []
    for s in cb:   # collapse a callback split into pieces by nesting
        if not cb_first or s[0] > cb_first[-1][1] + 1e-3:
            cb_first.append([s[0], s[1]])
        else:
            cb_first[-1][1] = max(cb_first[-1][1], s[1])
    tr0 = next(e["t0"] for e in ev["events"] if e["kind"] == "trace_start")
    in_trace = [e for e in dev if e["t0"] >= tr0]
    # The profiler's clock is relative and the meter's is the epoch, so one offset has to be
    # recovered from the device callbacks, which both clocks see. Pairing them in order
    # assumed one meter event per callback thunk: with a second stack swapped onto the card
    # both stacks' callbacks carry the same op_name scope
    # (`.../alphafold_iteration/evoformer/pure_callback`) while the meter times one class, so
    # the lists have different lengths and the pairing slid by whole rounds -- 52 s of spread
    # against the 11.7 ms of the single-swap run this was written for. Take the offset the
    # most pairs agree on: every candidate difference is a vote, and the densest 5 ms window
    # of votes wins. Unmatched thunks and unmatched meter events simply cast no winning vote.
    votes = sorted(d["t0"] - c[0] for d in in_trace for c in cb_first)
    best_i, best_n = 0, 0
    j = 0
    for i, v in enumerate(votes):
        while votes[j] < v - 5e-3:
            j += 1
        if i - j + 1 > best_n:
            best_n, best_i = i - j + 1, (i + j) // 2
    off = votes[best_i]
    agree = [v for v in votes if abs(v - off) <= 5e-3]
    offs = agree

    tr1 = next((e["t0"] for e in ev["events"] if e["kind"] == "trace_stop"), float("inf"))
    sgs = [e for e in ev["events"] if e["phase"] == "sequence_gradients" and e["t0"] >= tr0
           and e["t1"] <= tr1 and e["dt"] > 1.0]
    lc = [e for e in ev["events"] if e["phase"] == "lower_compile"]
    report = {"instrument": "jax.profiler thunk events + optimised HLO op_name",
              "hlo": hlo[0], "programs": {f"{k[0]}#{k[1]}": v for k, v in programs.items()},
              "offset_spread_ms": round((offs[-1] - offs[0]) * 1e3, 2),
              "offset_votes": {"agreeing": len(agree), "meter_device_events": len(in_trace),
                               "callback_thunks": len(cb_first)},
              "rounds": [],
              "rounds_without_thunks": []}
    for sg in sgs:
        w0, w1 = sg["t0"] - off, sg["t1"] - off
        inwin = [(max(a, w0), min(b, w1), op) for a, b, op in segs if b > w0 and a < w1]
        # A round can sit inside the trace window and still hold no thunk: the tracer is
        # started and stopped at a round boundary, so the first or last round it brackets can
        # be one whose XLA execution fell outside. Reported rather than crashed on, and never
        # averaged into the map.
        if not inwin:
            report["rounds_without_thunks"].append(
                {"round": sg.get("round"), "sequence_gradients_s": round(sg["dt"], 3)})
            continue
        wall = wall_split(inwin)
        # device landmarks inside this call, for the bcx-round time partition
        marks = sorted((e["t0"] - off, e["t1"] - off, e["phase"]) for e in dev
                       if sg["t0"] <= e["t0"] <= sg["t1"])
        def phase(t):
            k = sum(1 for m in marks if m[1] <= t)
            return ["host: before trunk #1", "host: between trunk calls",
                    "host: fwd tail + bwd entry", "host: JAX backward tail"][min(k, 3)]
        by = collections.defaultdict(lambda: [0.0, 0.0])
        by_phase = collections.defaultdict(lambda: collections.Counter())
        covered = 0.0
        unm = collections.Counter()
        detail = collections.Counter()
        inh = 0.0
        for i, (a, b, op) in enumerate(inwin):
            if op not in labels:
                unm[op.split(".")[0]] += wall[i]
            elif op in inherited:
                inh += wall[i]
            mod, dirn, sub = labels.get(op, ("unmapped HLO", "?", "-"))
            detail[f"{mod} [{dirn}] {sub}"] += wall[i]
            key = mod if mod.startswith("device") else f"{mod} [{dirn}]"
            by[key][0] += wall[i]
            by[key][1] += b - a
            covered += wall[i]
            if not mod.startswith("device"):
                by_phase[phase((a + b) / 2)][mod] += wall[i]
        lcs = sum(e["dt"] for e in lc if sg["t0"] - 60 <= e["t0"] <= sg["t0"] + 0.01
                  and e.get("round") == sg.get("round"))
        outside = sg["dt"] - covered
        first = min(a for a, _, _ in inwin)
        last = max(b for _, b, _ in inwin)
        rows = sorted(({"module": k, "wall_s": round(v[0], 3), "cpu_s": round(v[1], 3),
                        "share_of_sg": round(v[0] / sg["dt"], 4)} for k, v in by.items()),
                      key=lambda r: -r["wall_s"])
        report["rounds"].append({
            "round": sg.get("round"), "sequence_gradients_s": round(sg["dt"], 3),
            "lower_compile_s": round(lcs, 3),
            "outside_xla_s": round(outside, 3),
            "outside_xla_before_first_thunk_s": round(first - w0, 3),
            "outside_xla_after_last_thunk_s": round(w1 - last, 3),
            "outside_xla_gaps_inside_execution_s": round(outside - (first - w0) - (w1 - last), 3),
            "charged_by_enclosing_computation_s": round(inh, 3),
            "unmapped_by_opcode_s": {k: round(v, 3) for k, v in unm.most_common(12)},
            "rows": rows,
            "detail": {k: round(v, 3) for k, v in detail.most_common() if v > 0.05},
            "by_phase": {p: {m: round(s, 3) for m, s in c.most_common()}
                         for p, c in by_phase.items()},
            "unmapped_ops": sorted({op for _, _, op in inwin if op not in labels})[:40]})
    json.dump(report, open(f"{run_dir}/hostmap.json", "w"), indent=1)
    for r in report["rounds"]:
        print(f"round {r['round']}: sequence_gradients {r['sequence_gradients_s']} s, "
              f"outside XLA {r['outside_xla_s']} s (lower+compile {r['lower_compile_s']} s; "
              f"before first thunk {r['outside_xla_before_first_thunk_s']}, after last "
              f"{r['outside_xla_after_last_thunk_s']}, gaps {r['outside_xla_gaps_inside_execution_s']}); "
              f"charged by enclosing computation {r['charged_by_enclosing_computation_s']} s")
        print("  unmapped by opcode:", r["unmapped_by_opcode_s"])
        for k, v in r["detail"].items():
            print(f"    detail {v:7.3f} s  {k}")
        for row in r["rows"]:
            print(f"  {row['wall_s']:8.3f} s wall {row['cpu_s']:8.3f} s cpu "
                  f"{100 * row['share_of_sg']:5.1f}%  {row['module']}")
        for p, c in sorted(r["by_phase"].items()):
            print(f"  [{p}] " + ", ".join(f"{m} {s}" for m, s in list(c.items())[:6]))
    print("offset spread ms", report["offset_spread_ms"])


if __name__ == "__main__":
    main(sys.argv[1])
