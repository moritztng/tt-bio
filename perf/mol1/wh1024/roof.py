"""Per-family seconds, achieved TFLOP/s and share of the fold from one prof.py attribution fold.

    python perf/mol1/wh1024/roof.py OUT/prof [--peak 145]

FLOPs come from each call's input shape in sections.jsonl, counted as state/mol1/recon/census.py does (2 x MAC,
matmul-class only): a trimul on (1, N, N, c) is N^2 (6 c^2) + N^3 c MAC, and so on. Families without a shape rule
(glue, host transfers, diffusion internals) show seconds only. Times are a call's own seconds, as in table.py.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table import family, stage  # noqa: E402


def flops(fam, shp):
    if not shp or len(shp) < 3:
        return 0.0
    c = shp[-1]
    if fam == "trimul":
        N = shp[-2]
        return 2.0 * (N * N * 6 * c * c + N ** 3 * c)
    if fam == "triattn":  # heads = c // 32 at head dim 32, so heads x 32 = c
        N = shp[-2]
        return 2.0 * (N * N * (5 * c * c + c * (c // 32)) + 2 * N ** 3 * c)
    if fam in ("pair transition", "MSA transition"):
        return 2.0 * shp[-3] * shp[-2] * 3 * 4 * c * c
    if fam == "single transition":
        return 2.0 * shp[-2] * 3 * 4 * c * c
    if fam == "opm" and len(shp) == 4:
        D, N = shp[-3], shp[-2]
        return 2.0 * (D * N * c * 64 + D * N * N * 32 * 32 + N * N * 1024 * 448)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prof")
    ap.add_argument("--peak", type=float, default=145.0, help="bf16 TFLOP/s roof of the part")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.prof) / "sections.jsonl")]
    own, fl = defaultdict(float), defaultdict(float)
    pend = defaultdict(float)
    for r in rows:
        p, s = r["path"], r["s"]
        depth = p.count(">")
        mine = s - pend.pop(depth + 1, 0.0)
        pend[depth] += s
        k = (stage(p), family(p, r.get("shape")))
        own[k] += mine
        fl[k] += flops(k[1], r.get("shape"))
    wall = json.loads((Path(a.prof) / "summary.json").read_text())["wall_s"]
    print(f"| stage | family | s | share | TFLOP | TFLOP/s | of {a.peak:.0f} TF roof |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(own, key=lambda k: -own[k]):
        s, f = own[k], fl[k] / 1e12
        if s < 0.005 * wall:
            continue
        rate = f"{f / s:.1f} | {100 * f / s / a.peak:.0f} %" if f else "- | -"
        print(f"| {k[0]} | {k[1]} | {s:.1f} | {100 * s / wall:.1f} % | {f:.1f} | {rate} |")
    print(f"| all | attribution fold wall | {wall:.1f} | | {sum(fl.values()) / 1e12:.1f} | | |")


if __name__ == "__main__":
    main()
