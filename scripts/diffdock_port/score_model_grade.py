"""Grade the ASSEMBLED score network against upstream's own outputs on a real complex.

    TT_VISIBLE_DEVICES= PYTHONPATH=$PWD env/bin/python3.10 \
        scripts/diffdock_port/score_model_grade.py --complex 7A9E_R4W [--steps 0,9,18]

No reference environment and no upstream clone: `dfd-ref` captured a full 19-step run of the
released v1.1 checkpoint on PoseBusters complexes, recording the score network's own
``(tr, rot, tor)`` for all 10 poses at every step together with the pose that produced it
(``state/dfd/goldens/<id>__s0/{scores,trajectory}.npz``). Replaying those inputs through this
port grades the whole network -- embedders, both towers, the cross graph, 11 convolutions and
the three heads -- against upstream's numbers rather than against another approximation.

The bar is NOT a flat PCC. Every graded module below this one sits at fp32 rounding, ~1e-7,
and eleven convolutions compound that; the rollout is also chaotic in the pose, so the honest
target is per-output PCC >= 0.999 (D-dfd-orch-8) with the max relative error reported beside
it, and the torsion output held only to the torus table's own 1 % Monte Carlo spread.

Two differences from upstream that are structural and are reported, not hidden:

* upstream's ``torch_cluster`` neighbour lists cap a node at 32 neighbours; this port takes
  every neighbour inside the cutoff, so where the cap binds the port sees a superset.
* ESM-2 here is tt-bio's shared host encoder, graded separately against the card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

GOLDENS = Path("/home/moritz/.coworker/state/dfd/goldens")
DATA = Path("/home/moritz/scratch/dfd/data/posebusters_dl/sel")
CKPT = Path("/home/moritz/scratch/dfd/ckpt/extracted/score_model")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() < 2:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = b.abs().max()
    return float((a - b).abs().max() / scale) if scale > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", default="7A9E_R4W")
    ap.add_argument("--steps", default="0,9,18")
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-fixture", default=None,
                    help="also torch.save the graph, ESM-2 block, step inputs and upstream "
                         "scores, so a card-side grade replays them without the data tree")
    args = ap.parse_args()

    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import sampling as S
    from tt_bio.diffdock.esm import receptor_x_columns
    from tt_bio.diffdock.input import ComplexSpec, build_complex_graph
    from tt_bio.diffdock.score_model import HostScoreNetwork, load_released

    name = args.complex
    gdir = GOLDENS / f"{name}__s0"
    index = json.loads((gdir / "index.json").read_text())
    scores = np.load(gdir / "scores.npz", allow_pickle=True)
    traj = np.load(gdir / "trajectory.npz", allow_pickle=True)

    cfg = C.load(str(CKPT), old_model=False)
    sd = torch.load(CKPT / "best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, cfg)

    spec = ComplexSpec(name=name, protein_path=str(DATA / name / f"{name}_protein.pdb"),
                       ligand_description=str(DATA / name / f"{name}_ligand.sdf"))
    graph = build_complex_graph(spec, seed=index.get("seed", 0))
    emb = receptor_x_columns(graph.chain_sequences, on_device=False)
    net = HostScoreNetwork(model, lm_embeddings=emb)
    score = net.score_fn(graph)

    # The golden's index records only the recipe fields that change the trajectory; the rest
    # come from the shipped recipe, which is the one dfd-ref ran.
    recipe = S.SamplingConfig.from_recipe({**S.recipe(), **index["recipe"]})
    bounds = S.SigmaBounds(tr_sigma_min=cfg.tr_sigma_min, tr_sigma_max=cfg.tr_sigma_max,
                           rot_sigma_min=cfg.rot_sigma_min, rot_sigma_max=cfg.rot_sigma_max,
                           tor_sigma_min=cfg.tor_sigma_min, tor_sigma_max=cfg.tor_sigma_max)
    t_tr, t_rot, t_tor = recipe.schedules(bounds)

    out = {"complex": name, "steps": {}}
    fixture = {"complex": name, "graph": graph, "emb": emb, "steps": {}, "want": {}}
    for k in [int(s) for s in args.steps.split(",")]:
        # the capture flattens the batch: [B*N, 3] for B poses of one ligand
        flat = torch.as_tensor(traj[f"s{k:02d}_pos_in"]).float()
        n_poses = int(scores[f"s{k:02d}_tr"].shape[0])
        pos_in = flat.reshape(n_poses, -1, 3)[: args.poses]
        tr_s, rot_s, tor_s = bounds.sigmas(float(t_tr[k]), float(t_rot[k]), float(t_tor[k]))
        step = S.StepInput(
            lig_pos=pos_in, step=k, t_tr=float(t_tr[k]), t_rot=float(t_rot[k]),
            t_tor=float(t_tor[k]), tr_sigma=tr_s, rot_sigma=rot_s, tor_sigma=tor_s,
            crop_cutoff=None if cfg.crop_beyond is None else 3 * tr_s + cfg.crop_beyond)
        with torch.no_grad():
            tr, rot, tor = score(step)
        want = {k2: torch.as_tensor(scores[f"s{k:02d}_{k2}"])[: args.poses].float()
                for k2 in ("tr", "rot", "tor")}
        fixture["steps"][k], fixture["want"][k] = step, want
        got = {"tr": tr.reshape(want["tr"].shape), "rot": rot.reshape(want["rot"].shape),
               "tor": tor.reshape(want["tor"].shape) if want["tor"].numel() else tor}
        row = {k2: {"pcc": pcc(got[k2], want[k2]), "max_rel_err": rel(got[k2], want[k2])}
               for k2 in ("tr", "rot", "tor")}
        out["steps"][k] = row
        print(f"step {k:2d} tr_sigma {tr_s:7.3f}  " + "  ".join(
            f"{k2} pcc {row[k2]['pcc']:.6f} rel {row[k2]['max_rel_err']:.3e}"
            for k2 in ("tr", "rot", "tor")), flush=True)

    print(json.dumps(out))
    if args.save_fixture:
        torch.save(fixture, args.save_fixture)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
