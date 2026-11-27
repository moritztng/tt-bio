"""Dump upstream DiffDock-L's score-network INPUTS and OUTPUTS for one complex, pose 0.

This is the grade the orchestrator asked for: the assembled network against upstream's
``TensorProductScoreModel`` with the released checkpoint, on real complexes at several
sigmas.  It deliberately dumps the *inputs* too -- node features, positions, both static
edge lists, the per-step sigmas -- so the port can be fed the exact same graph.  That
separates the network's arithmetic from the graph builder: if the port matches here but
not on a replayed rollout, the fault is in ``build_complex_graph``, not in the model.

Reference environment only (needs torch_cluster / e3nn / torch_geometric):

    cd ~/scratch/dfd/upstream && PYTHONPATH=. ~/scratch/dfd/venv-ref/bin/python \\
      /home/moritz/.coworker/wt/dfd-parity/scripts/diffdock_port/score_net_parity.py \\
      --id 7A9E_R4W --steps 0,5,10,15,18 --out /tmp/dfd_net_7A9E.npz
"""
import argparse
import copy
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="7A9E_R4W")
    ap.add_argument("--steps", default="0,5,10,15,18",
                    help="forward-call indices; upstream makes actual_steps (19) calls, 0..18")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = [int(s) for s in a.steps.split(",")]

    L.setup(a.seed)
    from torch_geometric.loader import DataLoader
    from utils.diffusion_utils import get_t_schedule, t_to_sigma as t_to_sigma_compl
    from utils.inference_utils import InferenceDataset
    from utils.sampling import randomize_position, sampling
    from utils.utils import get_model

    cid = a.id
    prot, lig = PB / cid / f"{cid}_protein.pdb", PB / cid / f"{cid}_ligand.sdf"
    run_dir = D / "runs/net-parity" / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    inf = yaml.safe_load((D / "upstream/default_inference_args.yaml").read_text())
    with open(D / "ckpt/extracted/score_model/model_parameters.yml") as f:
        sma = Namespace(**yaml.full_load(f))

    ds = InferenceDataset(
        out_dir=str(run_dir), complex_names=[cid], protein_files=[str(prot)],
        ligand_descriptions=[str(lig)], protein_sequences=[None], lm_embeddings=True,
        receptor_radius=sma.receptor_radius, remove_hs=sma.remove_hs,
        c_alpha_max_neighbors=sma.c_alpha_max_neighbors, all_atoms=sma.all_atoms,
        atom_radius=sma.atom_radius, atom_max_neighbors=sma.atom_max_neighbors,
        precomputed_lm_embeddings=lm_embeddings_for(cid, str(prot)),
        knn_only_graph=(not sma.not_knn_only_graph) if hasattr(sma, "not_knn_only_graph") else False)

    device = torch.device("cpu")
    t_to_sigma = partial(t_to_sigma_compl, args=sma)
    model = get_model(sma, device, t_to_sigma=t_to_sigma, no_parallel=True,
                      old=inf.get("old_score_model", False))
    model.load_state_dict(torch.load(D / "ckpt/extracted/score_model" / inf["ckpt"],
                                     map_location="cpu", weights_only=False), strict=True)
    model.eval()

    n = inf["samples_per_complex"]
    graph = next(iter(DataLoader(dataset=ds, batch_size=1, shuffle=False)))
    assert bool(graph.success[0])
    data_list = [copy.deepcopy(graph) for _ in range(n)]
    sched = get_t_schedule(sigma_schedule=inf["sigma_schedule"],
                           inference_steps=inf["inference_steps"],
                           inf_sched_alpha=inf["inf_sched_alpha"],
                           inf_sched_beta=inf["inf_sched_beta"])

    flat, state = {}, {"step": -1}
    fwd = model.forward

    def forward(data):
        state["step"] += 1
        out = fwd(data)
        step = state["step"]
        if step in want:
            lb = data["ligand"].batch.numpy()
            rb = data["receptor"].batch.numpy()
            nl, nr = int((lb == 0).sum()), int((rb == 0).sum())
            be = data["ligand", "ligand"].edge_index.numpy()
            bm = be[0] < nl                      # pose-0 columns of the bond graph
            re = data["receptor", "receptor"].edge_index.numpy()
            rm = re[0] < nr
            em = data["ligand"].edge_mask.numpy()[bm]
            p = f"s{step}_"
            g = lambda t: t.detach().numpy()
            flat.update({
                p + "lig_x": g(data["ligand"].x)[:nl].astype(np.float32),
                p + "lig_pos": g(data["ligand"].pos)[:nl].astype(np.float32),
                p + "bond_index": be[:, bm].astype(np.int32),
                p + "bond_attr": g(data["ligand", "ligand"].edge_attr)[bm].astype(np.float32),
                p + "edge_mask": em.astype(np.bool_),
                p + "rec_x": g(data["receptor"].x)[:nr].astype(np.float32),
                p + "rec_pos": g(data["receptor"].pos)[:nr].astype(np.float32),
                p + "rec_edge_index": re[:, rm].astype(np.int32),
                p + "t_tr": np.asarray(g(data["ligand"].node_t["tr"])[0], np.float64),
                p + "t_rot": np.asarray(g(data["ligand"].node_t["rot"])[0], np.float64),
                p + "t_tor": np.asarray(g(data["ligand"].node_t["tor"])[0], np.float64),
                p + "tr": g(out[0])[0].astype(np.float64),
                p + "rot": g(out[1])[0].astype(np.float64),
                p + "tor": g(out[2])[:int(em.sum())].astype(np.float64),
            })
            if hasattr(data["receptor"], "lm_embedding"):
                flat[p + "rec_lm"] = g(data["receptor"].lm_embedding)[:nr].astype(np.float32)
        return out

    model.forward = forward
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    randomize_position(data_list, sma.no_torsion, False, sma.tr_sigma_max,
                       initial_noise_std_proportion=inf["initial_noise_std_proportion"])
    sampling(data_list=data_list, model=model, inference_steps=inf["actual_steps"],
             tr_schedule=sched, rot_schedule=sched, tor_schedule=sched, device=device,
             t_to_sigma=t_to_sigma, model_args=sma, confidence_model=None,
             confidence_data_list=None, confidence_model_args=None, batch_size=n,
             no_final_step_noise=inf["no_final_step_noise"],
             temp_sampling=[inf["temp_sampling_tr"], inf["temp_sampling_rot"], inf["temp_sampling_tor"]],
             temp_psi=[inf["temp_psi_tr"], inf["temp_psi_rot"], inf["temp_psi_tor"]],
             temp_sigma_data=[inf["temp_sigma_data_tr"], inf["temp_sigma_data_rot"],
                              inf["temp_sigma_data_tor"]])
    # `sampling` calls forward `actual_steps` times, so indices run 0..actual_steps-1 and a
    # `--steps` entry at or past that count captures nothing. Stop rather than write an
    # archive that is quietly short: a grade over fewer cells than were asked for reads as a
    # pass.
    got = sorted({int(k[1:].split("_")[0]) for k in flat if k.startswith("s") and k[1].isdigit()})
    missing = sorted(set(want) - set(got))
    if missing:
        sys.exit(f"requested steps {missing} were never reached: upstream made "
                 f"{state['step'] + 1} forward calls (steps 0..{state['step']})")
    flat["steps"] = np.asarray(got, np.int32)
    flat["complex"] = np.asarray(cid)
    np.savez(a.out, **flat)
    print("wrote", a.out, "steps", want)


if __name__ == "__main__":
    main()
