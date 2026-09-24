"""The rows view on the embedding models, which batch sequences into one [B, L, D] forward.

A served embed request of more than one sequence runs batches of up to 8 rows, so its
projections are rank-3 with B > 1 and the view reaches them. A single sequence is a batch of
one and the guard leaves it alone. So the single-sequence embedding is the reference the
batched rows already answer to (`esmc.embed_sequences` promises each row matches running it
alone), and both arms are graded against it, side by side.

    python3 perf/bcx_oplin/embed_ab.py --model esmc-300m --out perf/bcx_oplin/embed/esmc-300m.json
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common import Census, Clock, arm  # noqa: E402

CDK2 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLH"
        "QDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWY"
        "RAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVV"
        "PPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def compare(x, ref, seqs):
    return dict(per_residue_rel_l2_max=round(max(rel(x[k].per_residue, ref[k].per_residue) for k in seqs), 6),
                pooled_cos_min=round(min(cos(x[k].pooled, ref[k].pooled) for k in seqs), 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=16, help="sequences, windows of CDK2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.model.startswith("saprot"):
        from tt_bio import saprot as M
        m = M.load_saprot(a.model)
    else:
        from tt_bio import esmc as M
        m = M.load_esmc(a.model)
    # Windows of CDK2 from 64 to 298 aa, so the batches carry real padding.
    seqs = {f"w{i}": CDK2[(7 * i) % 40:][: 64 + (234 * i) // (a.n - 1)] for i in range(a.n)}

    def run(on, batch_size=8, census=None):
        with arm(on), (census or contextlib.nullcontext()), Clock() as clk:
            t = time.perf_counter()
            e = M.embed_sequences(m, seqs, batch_size=batch_size)
            t = time.perf_counter() - t
        return {x.id: x for x in e}, round(t, 4), clk.stats()

    census = Census()
    run(True, census=census)                        # cold: compiles the programs of both arms
    run(False)
    alone, _, _ = run(False, batch_size=1)
    arms, walls, clocks = {}, {True: [], False: []}, []
    for r in range(a.rounds):
        for on in ((False, True) if r % 2 == 0 else (True, False)):
            arms[on], t, c = run(on)
            walls[on].append(t)
            clocks.append(c)

    res = dict(
        model=a.model, lengths=sorted({len(v) for v in seqs.values()}),
        on_vs_off=compare(arms[True], arms[False], seqs),
        off_vs_alone=compare(arms[False], alone, seqs),
        on_vs_alone=compare(arms[True], alone, seqs),
        wall_off_s=walls[False], wall_on_s=walls[True],
        speedup=round(float(np.median(walls[False]) / np.median(walls[True])), 4),
        aiclk=dict(min=min(c["min"] for c in clocks), max=max(c["max"] for c in clocks),
                   pci=clocks[0]["pci"]),
        census=census.summary())
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "census"}), flush=True)


if __name__ == "__main__":
    main()
