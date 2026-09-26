#!/usr/bin/env python3
"""Map a profiled BindCraft 2 gradient round's host seconds onto AF2 modules.

Adopted byte-for-byte from `perf/bcx_p10_hostmap/attrib.py` on `wk/bcx-p10-hostmap`,
so the composed arm is attributed by the same instrument that attributed main and the
two tables can be read against each other.

INSTRUMENT. `jax.profiler` on the XLA CPU backend records one event per executed thunk,
named by its optimised-HLO instruction (`hlo_op`). The optimised-HLO dump carries each
instruction's `op_name`, the haiku/JAX name scope it was traced under, so a thunk maps to a
module and to a direction (`transpose(` in the scope is the backward). The device calls are
`pure_callback` custom-calls and are thunks too, which is what lines this profile up with
`perf/bcx_round/meter.py`'s wall clock.

Prior art, and the two bugs it already paid for:
* `perf/bcx_seam/hostmap.py` (`origin/wk/bcx-extramsa`) is where the exclusive-segment and
  wall-split treatment, the fusion vote and the enclosing-computation inheritance come from.
  Its module table matched a regex against the WHOLE op_name path, and its reading of the
  template embedding (3.78 s/round) was later shown to be ~4x high.
* `perf/bcx_tmplseam/parts.py` (`origin/wk/bcx-tmplseam`) found why: `fold_iteration` is a
  substring of `alphafold_iteration`, which prefixes every op in the model, so a substring
  match charged the whole of AlphaFold to the structure module. Matching is done here
  against the path's COMPONENTS, never as a substring, and the regression is asserted below.
* `perf/bcx_tmplseam/opname.py` found the other direction: voting an unlabelled instruction
  into the enclosing computation's majority raw `op_name` put 163 s of a 181 s round on one
  module. Inheritance here votes over the coarse LABEL, as `hostmap.py` does, and what it
  inherits is reported on its own line.

WHAT IT STILL LIES ABOUT:
* Time outside every thunk inside `sequence_gradients` is Python and JAX dispatch, not XLA.
  It is its own row, with BindCraft 2's per-call `lower().compile()` split out of it.
* A fusion carries one op_name while fusing ops from neighbouring scopes; it is charged to
  the most common label among its fused instructions.
* The profiler costs time. The rounds of the same run that ran outside the trace window are
  reported beside the profiled ones, so the instrument's cost is measured.
"""
import argparse
import collections
import glob
import json
import pathlib
import re
import statistics as st
import sys

# --- labels -----------------------------------------------------------------------------

RULES = [
    ("template embedding", {
        "template_embedding", "template_embedding_multimer", "template_pair_stack",
        "single_template_embedding", "single_template_embedding_multimer",
        "template_pointwise_attention", "template_single_embedding", "template_projection"}),
    ("extra-MSA stack", {"extra_msa_stack"}),
    ("structure module", {
        "structure_module", "fold_iteration", "invariant_point_attention",
        "multi_rigid_sidechain", "rigid_sidechain", "quat_affine"}),
    ("heads: pLDDT", {"predicted_lddt_head"}),
    ("heads: PAE/pTM", {"predicted_aligned_error_head"}),
    ("heads: distogram", {"distogram_head"}),
    ("heads: masked MSA", {"masked_msa_head"}),
    ("heads: experimentally resolved", {"experimentally_resolved_head"}),
    ("Evoformer (JAX side)", {"evoformer", "evoformer_iteration"}),
    ("embedder: recycling", {"prev_pos_linear", "prev_msa_first_row_norm", "prev_pair_norm"}),
    ("embedder: extra-MSA input", {"extra_msa_activations"}),
    ("embedder: MSA/pair init", {
        "preprocess_1d", "preprocess_msa", "left_single", "right_single", "pair_activiations",
        "position_activations", "single_activations", "relpos", "seq_channel", "msa_channel"}),
    ("AlphaFold: other", {"alphafold", "alphafold_iteration"}),
]

SEAM = "seam: device callback"
WRAP = re.compile(r"^[\w.]+\((.*)\)$")
OPNAME = re.compile(r'op_name="([^"]*)"')
SRCFILE = re.compile(r'source_file="([^"]*)"')
COMP = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")
INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=")
CALLBACK = re.compile(r"custom_call_target=\"[^\"]*(callback|ffi_python)[^\"]*\"")


def components(opn):
    """Path components, unwrapping `jit(...)`, `jvp(...)`, `transpose(...)` as it goes."""
    out = set()
    for c in opn.split("/"):
        while c:
            out.add(c)
            m = WRAP.match(c)
            if not m:
                break
            c = m.group(1)
    return out


def label_of(opn, src, is_callback):
    if is_callback:
        return SEAM
    s = components(opn)
    for name, want in RULES:
        if s & want:
            return name
    base = (src or "").rsplit("/", 1)[-1]
    return f"outside AF2: {base}" if base else "outside AF2: unlabelled"


_EVO = ("jit(sequence_design_loss)/jvp(jit(apply))/jit(apply_fn)/alphafold/alphafold_iteration/"
        "evoformer/__layer_stack_no_state_1/while/body/eval_jaxpr/evoformer_iteration/"
        "triangle_attention_starting_node/attention/bqhc,bkhc->bhqk/dot_general")
assert label_of(_EVO, "", False) == "Evoformer (JAX side)"
assert label_of(_EVO.replace("evoformer_iteration", "fold_iteration"), "", False) \
    == "structure module", "fold_iteration must not be read out of alphafold_iteration"
assert label_of("jit(sequence_design_loss)/alphafold/alphafold_iteration/x", "", False) \
    == "AlphaFold: other"


def sub_of(opn):
    """The AF2 op inside a module, as the last recognised component on the path."""
    subs = ("triangle_multiplication_outgoing", "triangle_multiplication_incoming",
            "triangle_attention_starting_node", "triangle_attention_ending_node",
            "pair_transition", "outer_product_mean", "msa_row_attention_with_pair_bias",
            "msa_column_global_attention", "msa_column_attention", "msa_transition",
            "template_pointwise_attention", "template_pair_stack",
            "invariant_point_attention", "rigid_sidechain", "transition")
    s = components(opn)
    got = [p for p in subs if p in s]
    return got[-1] if got else (opn.rsplit("/", 1)[-1] or "other")


# --- the HLO dump -----------------------------------------------------------------------

def parse_hlo(paths):
    """instruction -> (label, direction, sub). Fusions vote over the body they fuse."""
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
            im = INSTR.match(line)
            if not im:
                continue
            name = im.group(1)
            on = OPNAME.search(line)
            lab = None
            if on:
                opn = on.group(1)
                sf = SRCFILE.search(line)
                lab = (label_of(opn, sf.group(1) if sf else "",
                                bool(CALLBACK.search(line))),
                       "bwd" if "transpose(" in opn else "fwd", sub_of(opn))
                body[comp].append(lab)
            elif CALLBACK.search(line):
                lab = (SEAM, "fwd", "callback")
            own.setdefault(name, lab)
            where.setdefault(name, comp)
            cm = re.search(r"calls=%?([\w.\-]+)", line)
            if cm:
                calls[name] = cm.group(1)
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
    return out, inherited


# --- the trace --------------------------------------------------------------------------

def exclusive_segments(events):
    """Leaf-time segments on one thread: [(t0, t1, hlo_op)], nested time removed."""
    evs = sorted(events, key=lambda e: (e[0], -e[1]))
    segs, stack = [], []

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


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--module", default="jit_sequence_design_loss")
    args = ap.parse_args()
    run = args.run_dir

    from jax.profiler import ProfileData
    ev = json.load(open(f"{run}/round_events.json"))
    xp = sorted(glob.glob(f"{run}/trace/**/*.xplane.pb", recursive=True))
    assert xp, f"no trace under {run}/trace"
    pd = ProfileData.from_file(xp[-1])
    thunks = collections.defaultdict(list)
    programs = collections.Counter()
    for pl in pd.planes:
        for ln in pl.lines:
            for e in ln.events:
                stats = dict(e.stats)
                if "hlo_op" not in stats or e.name.startswith("end: "):
                    continue
                programs[(stats.get("hlo_module"), stats.get("program_id"))] += 1
                if stats.get("hlo_module") != args.module:
                    continue
                thunks[ln.name].append((e.start_ns / 1e9, e.end_ns / 1e9, stats["hlo_op"]))
    pid = max((p for p in programs if p[0] == args.module), key=lambda p: programs[p])[1]
    hlo = sorted(glob.glob(f"{run}/hlo/module_{int(pid):04d}.{args.module}"
                           ".cpu_after_optimizations.txt"))
    assert hlo, f"no optimised-HLO dump for program_id {pid} under {run}/hlo"
    labels, inherited = parse_hlo(hlo)
    segs = [s for line in thunks.values() for s in exclusive_segments(line)]

    # Align the profiler's relative clock to the meter's epoch clock on the device callbacks:
    # the same events, timed by both instruments.
    dev = sorted((e for e in ev["events"] if e["kind"] == "device"), key=lambda e: e["t0"])
    cb = sorted((s for s in segs if labels.get(s[2], ("",))[0] == SEAM), key=lambda s: s[0])
    merged = []
    for s in cb:
        if not merged or s[0] > merged[-1][1] + 1e-3:
            merged.append([s[0], s[1]])
        else:
            merged[-1][1] = max(merged[-1][1], s[1])
    tr0 = next(e["t0"] for e in ev["events"] if e["kind"] == "trace_start")
    tr1 = next((e["t0"] for e in ev["events"] if e["kind"] == "trace_stop"), float("inf"))
    in_trace = [e for e in dev if e["t0"] >= tr0]
    offs = sorted(d["t0"] - c[0] for d, c in zip(in_trace, merged))
    assert offs, "no device callback inside the trace window to align on"
    off = offs[len(offs) // 2]

    sgs = [e for e in ev["events"] if e["phase"] == "sequence_gradients"
           and e["t0"] >= tr0 and e["t1"] <= tr1 and e["dt"] > 1.0]
    lc = [e for e in ev["events"] if e["phase"] == "lower_compile"]
    rounds = []
    for sg in sgs:
        w0, w1 = sg["t0"] - off, sg["t1"] - off
        inwin = [(max(a, w0), min(b, w1), op) for a, b, op in segs if b > w0 and a < w1]
        wall = wall_split(inwin)
        by = collections.Counter()
        detail = collections.Counter()
        unmapped = collections.Counter()
        inh = covered = 0.0
        for i, (a, b, op) in enumerate(inwin):
            lab = labels.get(op)
            if lab is None:
                unmapped[op.split(".")[0]] += wall[i]
                lab = ("unmapped HLO", "?", "-")
            elif op in inherited:
                inh += wall[i]
            mod, dirn, sub = lab
            by[(mod, dirn)] += wall[i]
            detail[f"{mod} [{dirn}] {sub}"] += wall[i]
            covered += wall[i]
        device_meter = sum(e["dt"] for e in dev
                           if sg["t0"] <= e["t0"] and e["t1"] <= sg["t1"])
        seam_thunk = sum(v for (m, _), v in by.items() if m == SEAM)
        lcs = sum(e["dt"] for e in lc if sg["t0"] - 1 <= e["t0"] <= sg["t1"])
        first = min(a for a, _, _ in inwin)
        last = max(b for _, b, _ in inwin)
        rounds.append({
            "round": sg.get("round"), "sequence_gradients_s": round(sg["dt"], 3),
            "device_meter_s": round(device_meter, 3),
            "host_in_sg_s": round(sg["dt"] - device_meter, 3),
            "seam_thunk_s": round(seam_thunk, 3),
            "xla_covered_s": round(covered, 3),
            "outside_xla_s": round(sg["dt"] - covered, 3),
            "outside_xla_before_first_thunk_s": round(first - w0, 3),
            "outside_xla_after_last_thunk_s": round(w1 - last, 3),
            "lower_compile_s": round(lcs, 3),
            "inherited_label_s": round(inh, 3),
            "unmapped_s": {k: round(v, 3) for k, v in unmapped.most_common(10)},
            "modules": {f"{m}|{d}": round(v, 4) for (m, d), v in by.most_common()},
            "detail": {k: round(v, 3) for k, v in detail.most_common() if v > 0.02}})

    keys = sorted({k for r in rounds for k in r["modules"]})
    table = []
    for k in keys:
        xs = [r["modules"].get(k, 0.0) for r in rounds]
        mod, d = k.split("|")
        table.append({"module": mod, "dir": d, "med_s": med(xs),
                      "min_s": round(min(xs), 3), "max_s": round(max(xs), 3)})
    table.sort(key=lambda r: -r["med_s"])
    host_rows = [r for r in table if r["module"] != SEAM]
    host_sum = sum(r["med_s"] for r in host_rows)
    out_xla = med([r["outside_xla_s"] for r in rounds])
    host_med = med([r["host_in_sg_s"] for r in rounds])
    report = {
        "run": run, "hlo": hlo[0], "profiled_rounds": len(rounds),
        "align_offset_spread_ms": round((offs[-1] - offs[0]) * 1e3, 2),
        "median": {
            "sequence_gradients_s": med([r["sequence_gradients_s"] for r in rounds]),
            "device_meter_s": med([r["device_meter_s"] for r in rounds]),
            "host_in_sg_s": host_med,
            "seam_thunk_s": med([r["seam_thunk_s"] for r in rounds]),
            "xla_module_sum_s": round(host_sum, 3),
            "outside_xla_s": out_xla,
            "lower_compile_s": med([r["lower_compile_s"] for r in rounds]),
            "table_sum_s": round(host_sum + out_xla, 3),
            "residual_s": round(host_med - host_sum - out_xla, 3),
            "residual_pct_of_host": round(
                100 * (host_med - host_sum - out_xla) / host_med, 1) if host_med else None},
        "table": table, "rounds": rounds}
    pathlib.Path(run, "hostmap.json").write_text(json.dumps(report, indent=1))
    m = report["median"]
    print(json.dumps(m, indent=1))
    print(f"\n{'module':40s} {'dir':4s} {'med s':>8s} {'min':>8s} {'max':>8s}")
    for r in table:
        print(f"{r['module']:40s} {r['dir']:4s} {r['med_s']:8.3f} "
              f"{r['min_s']:8.3f} {r['max_s']:8.3f}")
    print(f"{'outside XLA (python + dispatch)':40s} {'-':4s} {out_xla:8.3f}")
    print(f"{'RESIDUAL vs measured host wall':40s} {'-':4s} "
          f"{m['residual_s']:8.3f}  ({m['residual_pct_of_host']} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
