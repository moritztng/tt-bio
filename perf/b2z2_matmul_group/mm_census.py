#!/usr/bin/env python3
"""The 387 Matmul programs of one diffusion step, bucketed by shape.

Reuses `b2z2-step-program-fusion`'s aligner (site_cost.py) for the ttnn call site and reads the
operand shapes straight off the same armed capture, so a bucket carries both what it costs and
which line of tenstorrent.py issued it.
"""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b2z2_step_fusion"))
from site_cost import load_device, split_reps, align, short  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "perf/b2z2_step_fusion/src/ops_perf_step_whglx_c2.csv.gz"
OPS = ROOT / "perf/b2z2_step_fusion/ops_base_wh_c2.json"


def shape(r, k):
    d = [r.get(f"INPUT_{k}_{ax}_PAD[LOGICAL]", "") for ax in "WZYX"]
    if not all(d) or d[0] == "":
        return None
    def g(v):
        v = v.strip()
        return int(v.split("[")[0]) if v else 0
    return tuple(g(x) for x in d)


def main():
    reps = split_reps(load_device(CSV))
    rep = reps[0]
    codes = [short(r["OP CODE"]) for r in rep]
    durs = [[float(rp[k]["DEVICE KERNEL DURATION [ns]"]) for rp in reps] for k in range(len(codes))]
    dur = [sorted(d)[len(d) // 2] for d in durs]
    ops = json.loads(OPS.read_text())["ops"]["sequence"]
    pairs, consumed = align(ops, codes)
    assert consumed == len(codes), (consumed, len(codes))
    ttnn_of_dev = {j: i for i, j in pairs}

    rows = []
    for j, c in enumerate(codes):
        if not c.startswith("Matmul"):
            continue
        r = rep[j]
        rows.append({
            "dev_i": j, "ttnn_i": ttnn_of_dev.get(j), "ttnn": ops[ttnn_of_dev[j]],
            "us": round(dur[j] / 1e3, 3),
            "a": shape(r, 0), "b": shape(r, 1),
            "out": shape(r, 2) if r.get("OUTPUT_0_W_PAD[LOGICAL]") is None else None,
            "a_dt": r.get("INPUT_0_DATATYPE", "").strip(),
            "b_dt": r.get("INPUT_1_DATATYPE", "").strip(),
            "a_mem": r.get("INPUT_0_MEMORY", "").strip(),
            "cores": int(r["CORE COUNT"]), "fid": r.get("MATH FIDELITY", "").strip(),
            "cbwait": r.get("DEVICE COMPUTE CB WAIT FRONT [ns]", "").strip(),
            "trisc1": r.get("DEVICE TRISC1 KERNEL DURATION [ns]", "").strip(),
            "attrs": r.get("ATTRIBUTES", "")[:400],
        })
    tot = sum(x["us"] for x in rows)
    print(f"{len(rows)} Matmul programs, {tot/1e3:.4f} ms")

    by = defaultdict(lambda: {"n": 0, "us": 0.0, "idx": []})
    for x in rows:
        key = (x["a"], x["b"], x["a_dt"], x["b_dt"], x["cores"])
        b = by[key]
        b["n"] += 1; b["us"] += x["us"]; b["idx"].append(x["dev_i"])
    print(f"{len(by)} distinct (A,B,dtypes,cores) buckets\n")
    print(f"{'n':>4} {'ms':>9} {'us/ea':>8} {'cores':>5}  A x B  dtypes")
    order = sorted(by.items(), key=lambda kv: -kv[1]["us"])
    for key, v in order:
        a, b, adt, bdt, cores = key
        print(f"{v['n']:>4} {v['us']/1e3:>9.4f} {v['us']/v['n']:>8.2f} {cores:>5}  "
              f"{a} x {b}  {adt}/{bdt}")
    out = ROOT / "perf/b2z2_matmul_group/mm_census_wh_c2.json"
    out.write_text(json.dumps({
        "n_matmul": len(rows), "total_ms": round(tot / 1e3, 4),
        "buckets": [{"a": k[0], "b": k[1], "a_dt": k[2], "b_dt": k[3], "cores": k[4],
                     "n": v["n"], "ms": round(v["us"] / 1e3, 4),
                     "us_each": round(v["us"] / v["n"], 3), "dev_idx": v["idx"]}
                    for k, v in order],
        "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
