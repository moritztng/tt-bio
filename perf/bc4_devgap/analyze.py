#!/usr/bin/env python3
"""Read a devprof.py record: device busy against card wall, the gap between programs, time by family.

    python perf/bc4_devgap/analyze.py _devgap/p1/devprof.json.gz [--json out.json]

All times come off the card's own clock (firmware/kernel start and end cycles of every program).
Gaps are measured only between consecutive programs of ONE profiler read: a read syncs the host,
so the gap that spans it is the instrument's. A gap is charged to the program that follows it
(its launch). Cycles convert to ns at each program's own ratio of reported ns to cycles.
"""
import argparse
import bisect
import collections
import gzip
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--json")
a = ap.parse_args()
doc = json.load(gzip.open(a.path, "rt"))
C = {c: i for i, c in enumerate(doc["program_cols"])}
INVALID = 2 ** 64 - 1

spans = sorted(doc["spans"])
starts = [s[0] for s in spans]


def name(rid):
    i = bisect.bisect_right(starts, rid) - 1
    if i >= 0 and spans[i][0] <= rid < spans[i][1]:
        return spans[i][2], spans[i][3]
    return "unmapped", "?"


# Family: what BC4's census calls a kernel family, from the ttnn verb and the tt_bio function that issued it.
RULES = [
    ("triangle multiplication", ("trimul", "triangle_mult", "tri_mul")),
    ("triangle attention", ("triatt", "triangle_att", "tri_att")),
    ("outer product mean", ("outer_product", "opm")),
    ("msa row attention", ("row_att", "msa_row")),
    ("msa column attention", ("col_att", "msa_col", "column")),
    ("transition", ("transition", "swiglu", "relu")),
    ("residual add", ("rne_add", "msa_add", "residual")),
    ("layer norm", ("layer_norm", "layernorm", "_ln")),
]


def family(label, origin):
    s = f"{label} {origin}".lower()
    for fam, keys in RULES:
        if any(k in s for k in keys):
            return fam
    if label in ("matmul", "linear", "experimental.minimal_matmul"):
        return "matmul (other)"
    if any(k in label for k in ("reshape", "permute", "transpose", "to_layout", "tilize", "concat", "slice",
                                "getitem", "clone", "copy", "typecast", "pad", "move", "to_memory_config")):
        return "data movement"
    return "elementwise/other"


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


out = dict(source=a.path, sha=doc["sha"], host=doc["host"], chip=doc["chip"], iterations=[])
for it in sorted({p[C["iteration"]] for p in doc["programs"]}):
    rows = [p for p in doc["programs"] if p[C["iteration"]] == it and p[C["fw_start_cyc"]] is not None]
    ratio = statistics.median(p[C["fw_ns"]] / (p[C["fw_end_cyc"]] - p[C["fw_start_cyc"]])
                              for p in rows if p[C["fw_end_cyc"]] > p[C["fw_start_cyc"]])   # ns per cycle
    batches = collections.defaultdict(list)
    for p in rows:
        batches[p[C["read"]]].append(p)
    fam_n, fam_k, fam_fw, fam_gap = (collections.Counter() for _ in range(4))
    lab_n, lab_k, lab_gap = collections.Counter(), collections.Counter(), collections.Counter()
    gaps, gaps_trace, gaps_eager, kgaps = [], [], [], []
    busy_fw = busy_k = span = 0.0
    n_trace = 0
    for _, b in sorted(batches.items()):
        b.sort(key=lambda p: p[C["fw_start_cyc"]])
        span += (max(p[C["fw_end_cyc"]] for p in b) - b[0][C["fw_start_cyc"]]) * ratio
        end = kend = None
        cur = None          # union of fw intervals
        kcur = None
        prev = None
        for p in b:
            s, e = p[C["fw_start_cyc"]], p[C["fw_end_cyc"]]
            ks, ke = p[C["k_start_cyc"]], p[C["k_end_cyc"]]
            lab, org = name(p[C["runtime_id"]])
            fam = family(lab, org)
            traced = p[C["trace_id"]] not in (None, INVALID)
            n_trace += traced
            fam_n[fam] += 1
            lab_n[lab] += 1
            fam_fw[fam] += p[C["fw_ns"]] or 0
            fam_k[fam] += p[C["k_ns"]] or 0
            lab_k[lab] += p[C["k_ns"]] or 0
            if end is not None:
                g = max(0.0, (s - end) * ratio)
                gaps.append(g)
                fam_gap[fam] += g
                lab_gap[lab] += g
                same = traced and prev[C["trace_id"]] == p[C["trace_id"]] and \
                    prev[C["trace_id_counter"]] == p[C["trace_id_counter"]]
                (gaps_trace if same else gaps_eager).append(g)
            if kend is not None and ks is not None:
                kgaps.append(max(0.0, (ks - kend) * ratio))
            # unions
            if cur is None or s > cur[1]:
                if cur:
                    busy_fw += (cur[1] - cur[0]) * ratio
                cur = [s, e]
            else:
                cur[1] = max(cur[1], e)
            if ks is not None:
                if kcur is None or ks > kcur[1]:
                    if kcur:
                        busy_k += (kcur[1] - kcur[0]) * ratio
                    kcur = [ks, ke]
                else:
                    kcur[1] = max(kcur[1], ke)
            end = e if end is None else max(end, e)
            kend = ke if ks is not None and (kend is None or ke > kend) else kend
            prev = p
        busy_fw += (cur[1] - cur[0]) * ratio
        if kcur:
            busy_k += (kcur[1] - kcur[0]) * ratio
    meta = next((m for m in doc["iterations"] if m["n"] == it), {})
    n = len(rows)
    rec = dict(
        iteration=it, programs=n, traced_programs=n_trace, batches=len(batches),
        mapped=sum(1 for p in rows if name(p[C["runtime_id"]])[0] != "unmapped") / max(n, 1),
        ns_per_cycle=ratio, host_wall_s=meta.get("wall_s"), read_s=meta.get("read_s"), aiclk=meta.get("aiclk"),
        load=[meta.get("load_start"), meta.get("load_end")],
        card_span_s=span / 1e9, fw_busy_s=busy_fw / 1e9, kernel_busy_s=busy_k / 1e9,
        kernel_sum_s=sum(p[C["k_ns"]] or 0 for p in rows) / 1e9, fw_sum_s=sum(p[C["fw_ns"]] or 0 for p in rows) / 1e9,
        gap_sum_s=sum(gaps) / 1e9, gaps=len(gaps),
        gap_us=dict(p10=pct(gaps, .1) / 1e3, p50=pct(gaps, .5) / 1e3, p90=pct(gaps, .9) / 1e3,
                    p99=pct(gaps, .99) / 1e3, mean=statistics.mean(gaps) / 1e3),
        gap_in_trace_us=dict(n=len(gaps_trace), p50=(pct(gaps_trace, .5) or 0) / 1e3,
                             mean=(statistics.mean(gaps_trace) if gaps_trace else 0) / 1e3,
                             sum_s=sum(gaps_trace) / 1e9),
        gap_eager_us=dict(n=len(gaps_eager), p50=(pct(gaps_eager, .5) or 0) / 1e3,
                          mean=(statistics.mean(gaps_eager) if gaps_eager else 0) / 1e3,
                          sum_s=sum(gaps_eager) / 1e9),
        kernel_gap_us=dict(p50=(pct(kgaps, .5) or 0) / 1e3, sum_s=sum(kgaps) / 1e9),
        families={f: dict(programs=fam_n[f], kernel_s=fam_k[f] / 1e9, fw_s=fam_fw[f] / 1e9, gap_s=fam_gap[f] / 1e9)
                  for f in sorted(fam_n, key=lambda f: -fam_n[f])},
        labels={l: dict(programs=lab_n[l], kernel_s=lab_k[l] / 1e9, gap_s=lab_gap[l] / 1e9)
                for l in sorted(lab_n, key=lambda l: -lab_n[l])[:40]},
    )
    out["iterations"].append(rec)
    print(f"\niteration {it}: {n} programs ({n_trace} traced, {rec['mapped']:.1%} named) in {len(batches)} reads; "
          f"host wall {rec['host_wall_s']:.3f} s (profiled, {rec['read_s']:.3f} s in reads), AICLK {rec['aiclk']}")
    print(f"  card span inside reads {rec['card_span_s']:.3f} s | fw busy {rec['fw_busy_s']:.3f} | kernel busy "
          f"{rec['kernel_busy_s']:.3f} | kernel sum {rec['kernel_sum_s']:.3f} | gap sum {rec['gap_sum_s']:.3f} s")
    g = rec["gap_us"]
    print(f"  gap us p10 {g['p10']:.2f} p50 {g['p50']:.2f} p90 {g['p90']:.2f} p99 {g['p99']:.2f} mean {g['mean']:.2f}; "
          f"in-trace n={rec['gap_in_trace_us']['n']} p50 {rec['gap_in_trace_us']['p50']:.2f} "
          f"sum {rec['gap_in_trace_us']['sum_s']:.3f} s; eager n={rec['gap_eager_us']['n']} "
          f"p50 {rec['gap_eager_us']['p50']:.2f} sum {rec['gap_eager_us']['sum_s']:.3f} s")
    print(f"  {'family':26} {'programs':>8} {'kernel s':>9} {'fw s':>8} {'gap s':>8}")
    for f, d in rec["families"].items():
        print(f"  {f:26} {d['programs']:8d} {d['kernel_s']:9.3f} {d['fw_s']:8.3f} {d['gap_s']:8.3f}")
    print(f"  {'ttnn verb':40} {'programs':>8} {'kernel s':>9} {'gap s':>8}")
    for l, d in list(rec["labels"].items())[:25]:
        print(f"  {l[:40]:40} {d['programs']:8d} {d['kernel_s']:9.3f} {d['gap_s']:8.3f}")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
