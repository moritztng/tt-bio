"""Grade the rollout against upstream's trajectory with the REAL score network.

    ~/rfd4-fair/env/bin/python scripts/diffdock_port/sampling_real_grade.py \
        --goldens ~/.coworker/state/dfd/goldens/8G43_ZU6__s0 \
        --ligand ~/scratch/dfd/data/posebusters_dl/sel/8G43_ZU6/8G43_ZU6_ligand.sdf

No upstream clone, no reference venv and no network forward pass: dfd-ref already captured a
full 19-step run of the released checkpoint on a PoseBusters complex, recording the score
network's own tr/rot/tor outputs at every step, every RNG draw in call order, and the pose in
and out of every ``modify_conformer_batch`` call. Replaying those is a stronger grade than
re-running the network would be, because it isolates this row: the scores are upstream's, so
anything that disagrees is the rollout.

Two halves, both over all 19 steps of the gated recipe:

* **the update** -- from upstream's score and upstream's recorded z draws, recompute the
  tr/rot/tor perturbation with our schedule, our g coefficients and our nine low-temperature
  constants, against what upstream applied. This is the sampler's arithmetic.
* **the move** -- feed upstream's recorded pose and perturbation through our
  ``modify_conformer`` against upstream's resulting pose. This is the rigid+torsion+Kabsch
  update on a real ligand with six rotatable bonds.

The draw record also pins the noise schedule: 20 setup draws then exactly 18 per-step triples
for 19 steps, because ``no_final_step_noise`` draws nothing on the last one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tt_bio.diffdock import sampling as S
from tt_bio.diffdock.config import SCORE_PIN


def sigma_bounds(params: dict | None = None) -> S.SigmaBounds:
    """The released score checkpoint's sigma bounds (``model_parameters.yml``)."""
    p = params or {}
    return S.SigmaBounds(
        tr_sigma_min=p.get("tr_sigma_min", 0.1), tr_sigma_max=p.get("tr_sigma_max", 19.0),
        rot_sigma_min=p.get("rot_sigma_min", 0.03), rot_sigma_max=p.get("rot_sigma_max", 1.55),
        tor_sigma_min=p.get("tor_sigma_min", 0.0314), tor_sigma_max=p.get("tor_sigma_max", 3.14))


def per_step_draws(draws: np.lib.npyio.NpzFile, n_poses: int, n_torsions: int, steps: int):
    """The per-step (tr_z, rot_z, tor_z) triples, found by shape in the recorded order.

    The record opens with the initial randomisation -- ``n_poses`` torsion uniforms and
    ``n_poses`` single translations -- and then runs triples. The scipy random rotation does
    not appear: ``Rotation.random`` does not go through the recorded numpy entry points.
    """
    names = list(draws["names"])
    arrays = [draws[f"d{i:05d}"] for i in range(len(names))]
    tail = arrays[2 * n_poses:]
    want = [(n_poses, 3), (n_poses, 3), (n_torsions * n_poses,)]
    triples = []
    for i in range(0, len(tail) - 2, 3):
        shapes = [tuple(tail[i + j].shape) for j in range(3)]
        if shapes != want:
            raise SystemExit(f"draw record does not line up at {i}: {shapes} != {want}")
        triples.append(tuple(torch.from_numpy(tail[i + j]) for j in range(3)))
    if len(triples) != steps - 1:
        raise SystemExit(f"{len(triples)} noisy steps recorded, expected {steps - 1} "
                         "(no_final_step_noise draws nothing on the last step)")
    return triples


def grade_updates(cfg, bounds, scores, recorded, triples, steps):
    """Our perturbation against upstream's, from upstream's scores and upstream's draws."""
    tr_sched, rot_sched, tor_sched = cfg.schedules(bounds)
    worst = {"tr": 0.0, "rot": 0.0, "tor": 0.0}
    for k in range(steps):
        last = k == steps - 1
        t_tr, t_rot, t_tor = float(tr_sched[k]), float(rot_sched[k]), float(tor_sched[k])
        dt_tr = float(tr_sched[k] - tr_sched[k + 1]) if not last else t_tr
        dt_rot = float(rot_sched[k] - rot_sched[k + 1]) if not last else t_rot
        dt_tor = float(tor_sched[k] - tor_sched[k + 1]) if not last else t_tor
        tr_sigma, rot_sigma, tor_sigma = bounds.sigmas(t_tr, t_rot, t_tor)
        tr_g, rot_g = bounds.g("tr", tr_sigma), bounds.g("rot", rot_sigma)
        tor_g = bounds.g("tor", tor_sigma)

        quiet = cfg.no_random or (cfg.no_final_step_noise and last)
        if quiet:
            tr_z = rot_z = torch.zeros(scores[k]["tr"].shape)
            tor_z = torch.zeros(scores[k]["tor"].shape)
        else:
            tr_z, rot_z, tor_z = triples[k]

        a_tr, w_tr = S._temperature_scale(0, cfg, tr_sigma, bounds, "tr")
        a_rot, w_rot = S._temperature_scale(1, cfg, rot_sigma, bounds, "rot")
        a_tor, w_tor = S._temperature_scale(2, cfg, tor_sigma, bounds, "tor")
        ours = {
            "tr": tr_g ** 2 * dt_tr * a_tr * scores[k]["tr"] + tr_g * np.sqrt(dt_tr * w_tr) * tr_z,
            "rot": rot_g ** 2 * dt_rot * a_rot * scores[k]["rot"] + rot_g * np.sqrt(dt_rot * w_rot) * rot_z,
            "tor": tor_g ** 2 * dt_tor * a_tor * scores[k]["tor"] + tor_g * np.sqrt(dt_tor * w_tor) * tor_z,
        }
        for key in worst:
            worst[key] = max(worst[key], float((ours[key] - recorded[k][key]).abs().max()))
    return worst


def grade_moves(recorded, edge_index, mask_rotate, n_poses):
    """Our conformer update against upstream's, step by step, on the real ligand."""
    worst = 0.0
    for step in recorded:
        pos_in = step["pos_in"].reshape(n_poses, -1, 3)
        out, _ = S.modify_conformer(pos_in, step["tr"], step["rot"], step["tor"],
                                    edge_index=edge_index, mask_rotate=mask_rotate)
        worst = max(worst, float((out.reshape(-1, 3) - step["pos_out"]).abs().max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True, help="a dfd-ref capture directory")
    ap.add_argument("--ligand", required=True, help="the complex's ligand SDF")
    ap.add_argument("--bar", type=float, default=1e-4, help="max abs deviation, Angstrom")
    args = ap.parse_args()

    gold = Path(args.goldens).expanduser()
    index = json.loads((gold / "index.json").read_text())
    traj = np.load(gold / "trajectory.npz")
    score_npz = np.load(gold / "scores.npz")
    draws = np.load(gold / "draws.npz", allow_pickle=True)

    steps = int(traj["n_steps"])
    recipe = index["recipe"]
    n_poses = int(recipe["samples_per_complex"])
    n_torsions = traj["s00_tor_update"].shape[0] // n_poses
    cfg = S.SamplingConfig.from_recipe(S.recipe(), **{
        k: recipe[k] for k in ("inference_steps", "actual_steps", "samples_per_complex",
                               "no_final_step_noise") if k in recipe})

    recorded = [{
        "pos_in": torch.from_numpy(traj[f"s{k:02d}_pos_in"]),
        "pos_out": torch.from_numpy(traj[f"s{k:02d}_pos_out"]),
        "tr": torch.from_numpy(traj[f"s{k:02d}_tr_update"]),
        "rot": torch.from_numpy(traj[f"s{k:02d}_rot_update"]),
        "tor": torch.from_numpy(traj[f"s{k:02d}_tor_update"]),
    } for k in range(steps)]
    scores = [{k: torch.from_numpy(score_npz[f"s{i:02d}_{k}"]) for k in ("tr", "rot", "tor")}
              for i in range(steps)]

    bounds = sigma_bounds({"tr_sigma_max": SCORE_PIN["tr_sigma_max"]})
    triples = per_step_draws(draws, n_poses, n_torsions, steps)
    worst_update = grade_updates(cfg, bounds, scores, recorded, triples, steps)

    from tt_bio.diffdock.ligand import build_ligand_graph, read_molecule

    mol = read_molecule(args.ligand, remove_hs=False)
    graph = build_ligand_graph(mol, remove_hs=True)
    edge_index = graph["edge_index"].T[graph["edge_mask"]]
    worst_move = grade_moves(recorded, edge_index, graph["mask_rotate"], n_poses)

    print(f"complex {index['complex']} seed {index['seed']}: {steps} steps, {n_poses} poses, "
          f"{n_torsions} rotatable bonds, {len(triples)} noisy steps recorded")
    print(f"  update  tr {worst_update['tr']:.3e}  rot {worst_update['rot']:.3e}  "
          f"tor {worst_update['tor']:.3e}   (worst abs, upstream's score and draws)")
    print(f"  move    {worst_move:.3e} A   (worst abs pose deviation over all steps)")
    ok = max(list(worst_update.values()) + [worst_move]) < args.bar
    print("PASS" if ok else "FAIL", f"(bar {args.bar:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
