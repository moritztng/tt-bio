"""Per-family table from two prof.py attribution folds: seconds at the small and the large size, their ratio, and the
ratio the family's work alone predicts.

    python perf/mol1/wh1024/table.py SMALL/prof LARGE/prof [--depth 4013 6865]

A family is the op a call lands in (trimul, triattn, transition, ...), placed in the stage that called it (the trunk
Pairformer's PairMixer or full blocks, the MSA module, the template stack, diffusion, confidence). Times are a leaf's
own seconds (its wrapped children subtracted), so the rows add up to the attribution fold's wall. The work ratio is the
family's scaling in padded tokens N and MSA depth D: N^3 for the triangle ops, N^2 for the pair-wise ops,
D*N^2 for OPM and PWA, D*N for the MSA transition.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

STAGES = (("msa_module", "MSA module"), ("template", "template"), ("pf_mixer", "trunk PairMixer"),
          ("pf_block", "trunk Pairformer"), ("pair_cond", "pair conditioning"), ("diffusion", "diffusion"),
          ("confidence", "confidence"), ("trunk", "trunk other"), ("fold", "fold other"))


def stage(path):
    parts = path.split(">")
    if "confidence_start" in parts or "confidence" in parts:
        return "confidence"
    for key, name in STAGES:
        if key in parts:
            return name
    return "outside fold"


def family(path, shape):
    leaf = path.rsplit(">", 1)[-1]
    if leaf == "transition" and shape:
        if len(shape) == 4 and shape[1] == shape[2]:
            return "pair transition"
        return "MSA transition" if len(shape) == 4 else "single transition"
    if leaf.startswith("xfer_") or leaf in ("host_unpark", "msa_embed", "opm_from_host", "msa_update_chunks"):
        return "host<->chip / MSA plumbing" if leaf.startswith(("xfer_", "host_")) else leaf
    if leaf in ("pf_mixer", "pf_block", "pairformer", "msa_module", "template", "trunk", "trunk_stage", "fold",
                "diffusion", "confidence", "confidence_start", "pair_cond"):
        return "glue (unwrapped ops)"
    return leaf


def work_exp(fam):
    return {"trimul": (3, 0), "triattn": (3, 0), "pair transition": (2, 0), "attn_pair_bias": (2, 0),
            "opm": (2, 1), "pwa": (2, 1), "MSA transition": (1, 1), "single transition": (1, 0)}.get(fam)


def load(d):
    """{(stage, family): own seconds} from sections.jsonl; a call's own time is its seconds minus its children's."""
    rows = [json.loads(l) for l in open(Path(d) / "sections.jsonl")]
    own = defaultdict(float)
    pend = defaultdict(float)  # lines are written at call exit, children before their parent: child seconds per depth
    for r in rows:
        p, s = r["path"], r["s"]
        depth = p.count(">")
        mine = s - pend.pop(depth + 1, 0.0)
        pend[depth] += s
        own[(stage(p), family(p, r.get("shape")))] += mine
    summ = json.loads((Path(d) / "summary.json").read_text())
    return own, summ["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("small")
    ap.add_argument("large")
    ap.add_argument("--n", type=int, nargs=2, default=(512, 1024), help="padded tokens, small and large")
    ap.add_argument("--depth", type=int, nargs=2, default=(4013, 6865), help="MSA rows, small and large")
    a = ap.parse_args()
    (A, wa), (B, wb) = load(a.small), load(a.large)
    rn, rd = a.n[1] / a.n[0], a.depth[1] / a.depth[0]
    print(f"| stage | family | {a.n[0]} s | {a.n[1]} s | time ratio | work ratio | share of {a.n[1]} |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(set(A) | set(B), key=lambda k: -B.get(k, 0)):
        x, y = A.get(k, 0.0), B.get(k, 0.0)
        if max(x, y) < 0.5:
            continue
        e = work_exp(k[1])
        wr = f"{rn ** e[0] * rd ** e[1]:.1f}" if e else "-"
        tr = f"{y / x:.1f}" if x else "-"
        print(f"| {k[0]} | {k[1]} | {x:.1f} | {y:.1f} | {tr} | {wr} | {100 * y / wb:.1f} % |")
    print(f"| all | attribution fold wall | {wa:.1f} | {wb:.1f} | {wb / wa:.1f} | | |")


if __name__ == "__main__":
    main()
