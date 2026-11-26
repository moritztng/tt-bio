"""Dump upstream's score-network INTERNALS for one pose, to localise an assembly gap.

``score_net_grade.py`` grades the port's three outputs on upstream's own graph tensors and
reports a gap; it cannot say WHICH layer first diverges.  This records, for pose 0 at a
given step: every graph the model builds inside ``forward`` (ligand radius edges, the cross
graph, the centre graph, the bond-conv graph) and the node features after every embedding
and convolution layer.  Replaying the port on the same inputs and comparing name by name
names the first divergent tensor.

Reference environment only:

    cd ~/scratch/dfd/upstream && PYTHONPATH=. ~/scratch/dfd/venv-ref/bin/python \\
      /home/moritz/.coworker/wt/dfd-parity/scripts/diffdock_port/score_net_internals.py \\
      --id 7A9E_R4W --steps 0 --out /tmp/dfd_int_7A9E.npz
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
    ap.add_argument("--steps", default="0")
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

    flat, state = {}, {"step": -1, "on": False}

    class _Done(Exception):
        """Raised to stop sampling once the last wanted step is recorded."""
    g = lambda t: t.detach().numpy()

    def rec(name, arr):
        if state["on"]:
            flat[f"s{state['step']}_{name}"] = arr

    # --- hooks on every layer that produces node features -------------------------------
    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            rec(name, g(o).astype(np.float32))
        return f

    for i, m in enumerate(model.rec_emb_layers):
        m.register_forward_hook(hook(f"rec_emb_{i}"))
    if hasattr(model, "lig_emb_layers"):
        for i, m in enumerate(model.lig_emb_layers):
            m.register_forward_hook(hook(f"lig_emb_{i}"))
    for i, m in enumerate(model.conv_layers):
        m.register_forward_hook(hook(f"conv_{i}"))
    model.final_conv.register_forward_hook(hook("final_conv"))
    model.tr_final_layer.register_forward_hook(hook("tr_final"))
    model.rot_final_layer.register_forward_hook(hook("rot_final"))
    model.tor_bond_conv.register_forward_hook(hook("tor_bond_conv"))
    model.tor_final_layer.register_forward_hook(hook("tor_final"))
    model.rec_node_embedding.register_forward_hook(hook("rec_node_emb"))
    model.rec_edge_embedding.register_forward_hook(hook("rec_edge_emb"))
    model.lig_node_embedding.register_forward_hook(hook("lig_node_emb"))
    model.lig_edge_embedding.register_forward_hook(hook("lig_edge_emb"))
    model.cross_edge_embedding.register_forward_hook(hook("cross_edge_emb"))
    model.rec_sigma_embedding.register_forward_hook(hook("rec_sigma_emb"))
    model.center_edge_embedding.register_forward_hook(hook("center_edge_emb"))
    model.final_edge_embedding.register_forward_hook(hook("final_edge_emb"))

    # --- wrap the graph builders ---------------------------------------------------------
    b_lig, b_cross = model.build_lig_conv_graph, model.build_cross_conv_graph
    b_center, b_bond = model.build_center_conv_graph, model.build_bond_conv_graph

    def lig_wrap(data):
        out = b_lig(data)
        rec("lig_edge_index", g(out[1]).astype(np.int32))
        rec("lig_edge_attr_in", g(out[2]).astype(np.float32))
        rec("lig_edge_sh", g(out[3]).astype(np.float32))
        return out

    def cross_wrap(data, cutoff):
        out = b_cross(data, cutoff)
        rec("cross_cutoff", np.asarray(
            g(cutoff) if torch.is_tensor(cutoff) else cutoff, np.float64))
        rec("lr_edge_index", g(out[0]).astype(np.int32))
        rec("lr_edge_sh", g(out[2]).astype(np.float32))
        return out

    def center_wrap(data):
        out = b_center(data)
        rec("center_edge_index", g(out[0]).astype(np.int32))
        rec("center_edge_sh", g(out[2]).astype(np.float32))
        return out

    def bond_wrap(data):
        out = b_bond(data)
        rec("tor_bonds", g(out[0]).astype(np.int32))
        rec("tor_edge_index", g(out[1]).astype(np.int32))
        rec("tor_edge_sh_pre", g(out[3]).astype(np.float32))
        return out

    model.build_lig_conv_graph = lig_wrap
    model.build_cross_conv_graph = cross_wrap
    model.build_center_conv_graph = center_wrap
    model.build_bond_conv_graph = bond_wrap

    fwd = model.forward

    def forward(data):
        state["step"] += 1
        # only pose 0 is dumped, and only on a wanted step; a batched forward still runs
        # all 10 poses, so every recorded tensor is sliced to pose 0's rows afterwards.
        state["on"] = state["step"] in want
        out = fwd(data)
        if state["on"]:
            lb, rb = data["ligand"].batch.numpy(), data["receptor"].batch.numpy()
            nl, nr = int((lb == 0).sum()), int((rb == 0).sum())
            be = data["ligand", "ligand"].edge_index.numpy()
            bm = be[0] < nl
            re = data["receptor", "receptor"].edge_index.numpy()
            rm = re[0] < nr
            em = data["ligand"].edge_mask.numpy()[bm]
            p = f"s{state['step']}_"
            flat.update({
                p + "n_lig": np.asarray(nl, np.int32),
                p + "n_rec": np.asarray(nr, np.int32),
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
        state["on"] = False
        if state["step"] >= max(want):
            raise _Done
        return out

    model.forward = forward
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    randomize_position(data_list, sma.no_torsion, False, sma.tr_sigma_max,
                       initial_noise_std_proportion=inf["initial_noise_std_proportion"])
    try:
        sampling(data_list=data_list, model=model, inference_steps=inf["actual_steps"],
                 tr_schedule=sched, rot_schedule=sched, tor_schedule=sched, device=device,
                 t_to_sigma=t_to_sigma, model_args=sma, confidence_model=None,
                 confidence_data_list=None, confidence_model_args=None, batch_size=n,
                 no_final_step_noise=inf["no_final_step_noise"],
                 temp_sampling=[inf["temp_sampling_tr"], inf["temp_sampling_rot"], inf["temp_sampling_tor"]],
                 temp_psi=[inf["temp_psi_tr"], inf["temp_psi_rot"], inf["temp_psi_tor"]],
                 temp_sigma_data=[inf["temp_sigma_data_tr"], inf["temp_sigma_data_rot"],
                                  inf["temp_sigma_data_tor"]])
    except _Done:
        pass
    flat["steps"] = np.asarray(want, np.int32)
    flat["complex"] = np.asarray(cid)
    np.savez(a.out, **flat)
    print("wrote", a.out, "keys", len(flat))


if __name__ == "__main__":
    main()
