"""The fixed-shape forward with every convolution on the card, graded against the host one.

    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEASE_CARDS=<chip> PYTHONPATH=<worktree> python3 \\
        scripts/diffdock_port/device_static_grade.py --fixture fx_8G43.pt --out device_static.json

`static_net.static_forward` is one shape for every pose and every step (D-dfd-device-3) and is graded
against the per-pose host model by `static_net_grade.py`. This runs the same function with
`convs_on_device`, so the only difference from that run is where the convolutions execute. Three
numbers per step: card vs the same static forward on the host in float32, every convolution against
its own float64 host layer, and the wall time with the AICLK sampled during it.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
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
    ap.add_argument("--poses", type=int, default=0, help="0: as many as the fixture carries")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    import tt_bio
    wt = os.environ.get("PYTHONPATH", "").split(":")[0]
    print("tt_bio from", tt_bio.__file__, flush=True)
    if wt and not tt_bio.__file__.startswith(wt):
        raise SystemExit(f"tt_bio resolved outside {wt}")

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import device as ddev
    from tt_bio.diffdock.capture import clock_summary, sample_aiclk
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.score_model import HostScoreNetwork, load_released
    from tt_bio.diffdock.static_net import StaticComplex, static_forward
    from tt_bio.tenstorrent import get_device

    fx = torch.load(args.fixture, weights_only=False)
    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    host = load_released(sd, cfg)
    card = load_released(sd, cfg)
    graph = fx["graph"]
    rec_x = HostScoreNetwork(host, lm_embeddings=fx["emb"])._receptor_x(graph.receptor)

    dev = get_device()
    convs = ddev.convs_on_device(card, dev, dtype=ttnn.bfloat16 if args.bf16 else None,
                                 grade=True)
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_aiclk, daemon=True,
                          args=(stop, samples, os.environ.get("TT_VISIBLE_DEVICES", "0")))
    th.start()

    out = {"complex": fx["complex"], "convs_on_device": len(convs),
           "dtype": "bfloat16" if args.bf16 else "float32", "steps": {}}
    for k, step in sorted(fx["steps"].items()):
        P = args.poses or step.lig_pos.shape[0]
        n_lig = step.lig_pos.shape[1]
        c = StaticComplex.build(graph.ligand, rec_x, graph.receptor["pos"].float(),
                                graph.receptor["edge_index"], P)
        lig_pos = pad_nodes(step.lig_pos[:P].reshape(P * n_lig, 3).float(), c.sg.n_lig_padded)
        sig = (step.t_tr, step.tr_sigma, step.rot_sigma, step.tor_sigma)
        with torch.no_grad():
            h = static_forward(host, c, lig_pos, *sig, crop_cutoff=step.crop_cutoff)
            for m in convs.values():
                m.log.clear()
            t0 = time.perf_counter()
            g = static_forward(card, c, lig_pos, *sig, crop_cutoff=step.crop_cutoff)
            secs = time.perf_counter() - t0
        row = {"poses": P, "seconds_untraced_with_grading": secs,
               "edges": {"lig": int(c.sg.lig_edge_index.shape[1]),
                         "cross": int(c.sg.cross_edge_index.shape[1]),
                         "rec": int(c.rec_edge_index.shape[1]),
                         "tor": int(c.sg.tor_edge_index.shape[1])}}
        for name, t in zip(("tr", "rot", "tor"), range(3)):
            row[name] = grade(g[t].reshape(-1), h[t].reshape(-1))
        worst = {}
        for name, m in convs.items():
            pc = [e["pcc"] for e in m.log if e["pcc"] is not None]
            worst[name] = dict(calls=len(m.log), min_pcc=min(pc) if pc else None,
                               max_rel_err=max((e["max_rel_err"] for e in m.log), default=None))
        row["convs"] = worst
        out["steps"][k] = row
        print(f"step {k}: " + "  ".join(
            f"{n} pcc {row[n].get('pcc')} rel {row[n].get('max_rel_err'):.3e}"
            for n in ("tr", "rot", "tor") if row[n])
            + f"  worst conv pcc {min(v['min_pcc'] for v in worst.values() if v['min_pcc'])}"
            + f"  {secs:.1f} s", flush=True)

    stop.set()
    th.join(timeout=30)
    out.update(chip=os.environ.get("TT_VISIBLE_DEVICES"), **clock_summary(samples))
    print(json.dumps({k: v for k, v in out.items() if k != "steps"}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
