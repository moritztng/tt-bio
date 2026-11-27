"""Upstream and the port on the SAME poses, over the pose space the rollout actually visits.

Four PoseBusters complexes die in the Kabsch SVD because the port's tr/rot heads return a
finite ~1e15 (D-dfd-orch-87).  Chasing the single trajectory that got there is fragile: the
seed conformer is environment-dependent (`ligand_env_fingerprint.py` measures it, and
`state/dfd/NAN-2026-11-27.md` §1 measured the same thing), so two venvs do not even visit the
same poses and "upstream stayed O(10)" was measured on upstream's poses, not on ours.

This removes the trajectory from the question.  It builds ONE graph per complex in upstream's
own `InferenceDataset`, then draws N poses from the sampler's own prior across the whole sigma
schedule, and runs **both implementations on the identical tensors in one process** -- upstream's
unmodified `TensorProductScoreModel` and the port's `ScoreModel`, both with the released v1.1
weights.  A pose where upstream is bounded and the port is not is a parity break in the port.
A pose where both blow up is the network's own behaviour on a pathological input and the port
is not the thing that is wrong.

Both the PCC and the RATIO are reported.  PCC alone is scale-invariant: it reads 1.000000
straight through a multiplicative gain defect, which is exactly the shape a bounded upstream
beside a 1e15 port would have.

    cd ~/scratch/dfd/upstream && PYTHONPATH=. ~/scratch/dfd/venv-ref/bin/python \\
      ~/.coworker/wt/dfd-parity/scripts/diffdock_port/pose_sweep_parity.py \\
      --id 7ZCC_OGA --poses 64 --out ~/scratch/dfd/parity-blowup/sweep_7ZCC_OGA.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from argparse import Namespace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

D = Path.home() / "scratch/dfd"
sys.path.insert(0, str(D / "scripts"))
import dfd_ref_lib as L  # noqa: E402
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_dfd_dump_graph_golden", Path(__file__).resolve().parent / "dump_graph_golden.py")
_dg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dg)
PB, lm_embeddings_for = _dg.PB, _dg.lm_embeddings_for

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: The four complexes that die in the Kabsch SVD (D-dfd-orch-87).
BLOWUP = ["7AVI_S2Q", "7TBU_S3P", "7X5N_5M5", "7ZCC_OGA"]


def pcc(a, b) -> float:
    a = torch.as_tensor(np.asarray(a)).flatten().double()
    b = torch.as_tensor(np.asarray(b)).flatten().double()
    if a.numel() < 2 or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def absmax(t) -> float:
    t = torch.as_tensor(np.asarray(t))
    f = t[torch.isfinite(t)]
    return float(f.abs().max()) if f.numel() else float("inf")


def random_rotation(rng) -> np.ndarray:
    """A uniform rotation, upstream's own construction (scipy Rotation.random)."""
    from scipy.spatial.transform import Rotation

    return Rotation.random(random_state=rng.integers(0, 2**31 - 1)).as_matrix()


def place(pos: np.ndarray, centre: np.ndarray, tr_sigma: float, rng) -> np.ndarray:
    """A pose from the sampler's own prior: random orientation, Gaussian translation."""
    local = pos - pos.mean(0, keepdims=True)
    return local @ random_rotation(rng).T + centre + rng.normal(0.0, tr_sigma, size=3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="7ZCC_OGA")
    ap.add_argument("--poses", type=int, default=64, help="poses per sigma level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio-bar", type=float, default=2.0,
                    help="flag a pose whose port/upstream head ratio exceeds this")
    ap.add_argument("--pcc-bar", type=float, default=0.999)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L.setup(a.seed)
    from torch_geometric.loader import DataLoader
    from utils.diffusion_utils import (get_t_schedule, set_time,
                                       t_to_sigma as t_to_sigma_compl)
    from utils.inference_utils import InferenceDataset
    from utils.utils import get_model

    cid = a.id
    prot = PB / cid / f"{cid}_protein.pdb"
    lig = str(PB / cid / f"{cid}_ligand.sdf")
    run_dir = D / "runs/pose-sweep" / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    inf = yaml.safe_load((D / "upstream/default_inference_args.yaml").read_text())
    with open(D / "ckpt/extracted/score_model/model_parameters.yml") as f:
        sma = Namespace(**yaml.full_load(f))

    ds = InferenceDataset(
        out_dir=str(run_dir), complex_names=[cid], protein_files=[str(prot)],
        ligand_descriptions=[lig], protein_sequences=[None], lm_embeddings=True,
        receptor_radius=sma.receptor_radius, remove_hs=sma.remove_hs,
        c_alpha_max_neighbors=sma.c_alpha_max_neighbors, all_atoms=sma.all_atoms,
        atom_radius=sma.atom_radius, atom_max_neighbors=sma.atom_max_neighbors,
        precomputed_lm_embeddings=lm_embeddings_for(cid, str(prot)),
        knn_only_graph=(not sma.not_knn_only_graph) if hasattr(sma, "not_knn_only_graph")
        else False)

    graph = next(iter(DataLoader(dataset=ds, batch_size=1, shuffle=False)))
    assert bool(graph.success[0]), f"{cid}: upstream could not build the graph"

    t_to_sigma = partial(t_to_sigma_compl, args=sma)
    up = get_model(sma, torch.device("cpu"), t_to_sigma=t_to_sigma, no_parallel=True,
                   old=inf.get("old_score_model", False))
    up.load_state_dict(torch.load(D / "ckpt/extracted/score_model" / inf["ckpt"],
                                  map_location="cpu", weights_only=False), strict=True)
    up.eval()

    # the port, same released weights, same process
    from tt_bio.diffdock import checkpoints, config as dfd_config, sampling as S, score_model
    ckpt, params = checkpoints.resolve("score", str(D / "ckpt/extracted/score_model"), None)
    cfg = dfd_config.load(params.parent, old_model=False, name="diffdock")
    port = score_model.load_released(
        torch.load(ckpt, map_location="cpu", weights_only=False), cfg).eval()
    bounds = S.SigmaBounds(tr_sigma_min=cfg.tr_sigma_min, tr_sigma_max=cfg.tr_sigma_max,
                           rot_sigma_min=cfg.rot_sigma_min, rot_sigma_max=cfg.rot_sigma_max,
                           tor_sigma_min=cfg.tor_sigma_min, tor_sigma_max=cfg.tor_sigma_max)

    sched = get_t_schedule(sigma_schedule=inf["sigma_schedule"],
                           inference_steps=inf["inference_steps"],
                           inf_sched_alpha=inf["inf_sched_alpha"],
                           inf_sched_beta=inf["inf_sched_beta"])

    rec_pos = graph["receptor"].pos.numpy()
    centre = rec_pos.mean(0)
    lig_pos0 = graph["ligand"].pos.numpy().copy()
    rng = np.random.default_rng(a.seed)

    rows, flagged = [], []
    for t in sched:
        tr_s, rot_s, tor_s = t_to_sigma(t, t, t)
        for k in range(a.poses):
            pos = place(lig_pos0, centre, float(tr_s), rng)

            g = copy.deepcopy(graph)
            g["ligand"].pos = torch.from_numpy(pos).float()
            # upstream's own time-setting, so the port is graded against upstream's convention
            set_time(g, t, t, t, t, 1, sma.all_atoms, torch.device("cpu"))
            with torch.no_grad():
                u_tr, u_rot, u_tor = up(g)[:3]

            nl = int(g["ligand"].num_nodes)
            nr = int(g["receptor"].num_nodes)
            be = g["ligand", "ligand"].edge_index.numpy()
            re_ = g["receptor", "receptor"].edge_index.numpy()
            em = g["ligand"].edge_mask.numpy()
            p_tr_s, p_rot_s, p_tor_s = bounds.sigmas(t, t, t)
            args = (g["ligand"].x.long(), torch.from_numpy(pos).float(),
                    torch.from_numpy(be).long(), g["ligand", "ligand"].edge_attr.float(),
                    torch.from_numpy(em), g["receptor"].x.float(),
                    g["receptor"].pos.float(), torch.from_numpy(re_).long(),
                    float(t), torch.as_tensor(p_tr_s), torch.as_tensor(p_rot_s),
                    torch.as_tensor(p_tor_s))
            with torch.no_grad():
                p_tr, p_rot, p_tor = port(*args)[:3]

            u = {"tr": np.asarray(u_tr[0].detach()), "rot": np.asarray(u_rot[0].detach()),
                 "tor": np.asarray(u_tor.detach())[:int(em.sum())]}
            p = {"tr": np.asarray(p_tr[0].detach()), "rot": np.asarray(p_rot[0].detach()),
                 "tor": np.asarray(p_tor.detach())[:int(em.sum())]}
            row = {"t": float(t), "pose": k, "n_lig": nl, "n_rec": nr,
                   "tr_sigma": float(tr_s),
                   "centre_dist": float(np.linalg.norm(pos.mean(0) - centre))}
            for head in ("tr", "rot", "tor"):
                ua, pa = absmax(u[head]), absmax(p[head])
                row[head] = {
                    "upstream_absmax": ua, "port_absmax": pa,
                    "ratio": (pa / ua) if ua > 0 and np.isfinite(ua) else float("nan"),
                    "pcc": pcc(u[head], p[head]),
                    "upstream_nonfinite": int((~np.isfinite(u[head])).sum()),
                    "port_nonfinite": int((~np.isfinite(p[head])).sum()),
                }
            rows.append(row)
            bad = [h for h in ("tr", "rot") if
                   (not np.isfinite(row[h]["ratio"]))
                   or row[h]["ratio"] > a.ratio_bar
                   or (np.isfinite(row[h]["pcc"]) and row[h]["pcc"] < a.pcc_bar)
                   or row[h]["port_nonfinite"] != row[h]["upstream_nonfinite"]]
            if bad:
                flagged.append({**row, "bad_heads": bad})
            print(f"t={t:.4f} pose {k:3d} "
                  + " | ".join(f"{h} up {row[h]['upstream_absmax']:.4g} port "
                               f"{row[h]['port_absmax']:.4g} ratio {row[h]['ratio']:.4f} "
                               f"pcc {row[h]['pcc']:.9f}" for h in ("tr", "rot")),
                  flush=True)

    def worst(key, head):
        vals = [r[head][key] for r in rows if np.isfinite(r[head][key])]
        return (max(vals) if key == "ratio" else min(vals)) if vals else float("nan")

    report = {
        "complex": cid, "seed": a.seed, "poses_per_sigma": a.poses,
        "n_poses": len(rows), "sigma_levels": [float(t) for t in sched],
        "bars": {"ratio": a.ratio_bar, "pcc": a.pcc_bar},
        "worst": {h: {"ratio": worst("ratio", h), "pcc": worst("pcc", h)}
                  for h in ("tr", "rot", "tor")},
        "upstream_max_absmax": {h: max(r[h]["upstream_absmax"] for r in rows)
                                for h in ("tr", "rot", "tor")},
        "port_max_absmax": {h: max(r[h]["port_absmax"] for r in rows)
                            for h in ("tr", "rot", "tor")},
        "n_flagged": len(flagged),
        "flagged": flagged[:40],
        "rows": rows,
    }
    Path(a.out).expanduser().write_text(json.dumps(report, indent=1))
    print(f"\n{cid}: {len(rows)} poses, {len(flagged)} flagged. "
          f"worst tr ratio {report['worst']['tr']['ratio']:.4f}, "
          f"worst tr pcc {report['worst']['tr']['pcc']:.9f}. "
          f"upstream max |tr| {report['upstream_max_absmax']['tr']:.4g}, "
          f"port max |tr| {report['port_max_absmax']['tr']:.4g}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
