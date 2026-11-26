"""Grade the fixed-shape forward against the per-pose host model, card-free.

    PYTHONPATH=<worktree> python3 scripts/diffdock_port/static_net_grade.py \
        --fixture /tmp/dfd_fx_8G43.pt --out static.json

`static_net.static_forward` runs every pose of a step over `StaticGraphs`' complete edge sets with
the step's graphs as masks; `HostScoreNetwork.score_fn` runs upstream's per-pose forward with
`nonzero`-built graphs. Same submodules and the same weights, so any difference is the edge sets:
the bar is float32 rounding, PCC >= 0.999999.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

CKPT = Path(os.environ.get("DFD_CKPT", str(Path.home() / "scratch/dfd/ckpt/extracted")))


def grade(got, ref):
    got, ref = got.double().flatten(), ref.double().flatten()
    if ref.numel() == 0:
        return {}
    return dict(pcc=float(torch.corrcoef(torch.stack([got, ref]))[0, 1]) if ref.numel() > 1
                else None,
                max_rel_err=float((got - ref).abs().max() / ref.abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    import tt_bio
    print("tt_bio from", tt_bio.__file__, flush=True)
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.score_model import HostScoreNetwork, load_released
    from tt_bio.diffdock.static_net import StaticComplex, static_forward

    fx = torch.load(args.fixture, weights_only=False)
    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, cfg)
    host = HostScoreNetwork(model, lm_embeddings=fx["emb"])
    graph = fx["graph"]
    score = host.score_fn(graph)
    rec_x = host._receptor_x(graph.receptor)

    out = {"complex": fx["complex"], "steps": {}}
    for k, step in sorted(fx["steps"].items()):
        P, n_lig = step.lig_pos.shape[0], step.lig_pos.shape[1]
        with torch.no_grad():
            t0 = time.perf_counter()
            h = dict(zip(("tr", "rot", "tor"), score(step)))
            t_host = time.perf_counter() - t0
            c = StaticComplex.build(graph.ligand, rec_x, graph.receptor["pos"].float(),
                                    graph.receptor["edge_index"], P)
            lig_pos = pad_nodes(step.lig_pos.reshape(P * n_lig, 3).float(),
                                c.sg.n_lig_padded)
            t0 = time.perf_counter()
            s = dict(zip(("tr", "rot", "tor"),
                         static_forward(model, c, lig_pos, step.t_tr, step.tr_sigma,
                                        step.rot_sigma, step.tor_sigma, step.crop_cutoff)))
            t_static = time.perf_counter() - t0
        row = {"poses": P, "seconds_per_pose_host": t_host / P, "seconds_static": t_static,
               "edges": {"lig": int(c.sg.lig_edge_index.shape[1]),
                         "cross": int(c.sg.cross_edge_index.shape[1]),
                         "rec": int(c.rec_edge_index.shape[1]),
                         "tor": int(c.sg.tor_edge_index.shape[1])}}
        for n in ("tr", "rot", "tor"):
            row[n] = grade(s[n].reshape(-1), h[n].reshape(-1))
        out["steps"][k] = row
        print(f"step {k}: " + "  ".join(
            f"{n} pcc {row[n].get('pcc')} rel {row[n].get('max_rel_err'):.3e}"
            for n in ("tr", "rot", "tor") if row[n]), flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
