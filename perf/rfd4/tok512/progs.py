"""Token-side programs of a census profile, one row per (family, op, shapes, grid, fidelity), against its floor.

    python3 perf/rfd4/tok512/progs.py <census dir> [out.json]

Same attribution, per-design weighting and floor as land10/top_ops.py. The token side is every family that is not
a pair-tensor program (triangle, pair transition, pair write, pair bias, casts of z).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "land10"))
from top_ops import passes, shapes, table  # noqa: E402

TOKEN = ("attention", "token_transition", "adaln", "ada_scale", "residual", "recycle", "pair_factors", "pass_glue",
         "cast")


def config(r):
    a = r.get("ATTRIBUTES", "")
    m = re.search(r"(Matmul\w*ProgramConfig|SDPAProgramConfig)", a)
    g = re.search(r"compute_with_storage_grid_size=\(x=(\d+);y=(\d+)\)", a)
    q = re.search(r"q_chunk_size=(\d+)", a)
    k = re.search(r"k_chunk_size=(\d+)", a)
    parts = [m.group(1).replace("ProgramConfig", "") if m else "", f"{g.group(1)}x{g.group(2)}" if g else "",
             f"q{q.group(1)}k{k.group(1)}" if q and k else ""]
    return "/".join(p for p in parts if p)


def token(ops):
    # a cast of the [n, n, 384] pair tensor is pair side
    return [(f, r) for f, r in ops if f in TOKEN and not (f == "cast" and r.get("INPUT_0_Y_PAD[LOGICAL]", "").startswith(
        r.get("INPUT_0_Z_PAD[LOGICAL]", "-")) and int((r.get("INPUT_0_X_PAD[LOGICAL]") or "0").split("[")[-1].rstrip("]") or 0) == 384)]


def main():
    prof = sys.argv[1]
    run = json.load(open(f"{prof}/run.json"))
    first, rec = passes(prof)
    first, rec = token(first), token(rec)
    rows = table(first, rec, lambda f, r: (f, r["OP CODE"], " | ".join(
        (config(r), r.get("MATH FIDELITY", ""), shapes(r)))), run.get("batch", 1))
    tot = sum(x["device_s_per_design"] for x in rows)
    fl = sum(x["floor_s_per_design"] for x in rows)
    print(f"token side n={run['n']} b={run.get('batch', 1)} AICLK {run.get('aiclk')}: {tot:.3f} s/design, floor {fl:.3f}")
    for x in rows:
        if x["device_s_per_design"] < 0.004:
            continue
        print(f"{x['family']:16s} {x['op'][:24]:24s} {x['device_s_per_design']:6.3f} fl {x['floor_s_per_design']:6.3f}"
              f" x{x['x_floor']} n{x['ops_per_recycled_pass']} c{','.join(x['cores'])} | {x['shapes'][:170]}")
    if len(sys.argv) > 2:
        json.dump({"run": run, "token_s_per_design": round(tot, 3), "floor_s_per_design": round(fl, 3), "rows": rows},
                  open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
