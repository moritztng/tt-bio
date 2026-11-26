"""Does UPSTREAM DiffDock-L also explode when the receptor's ESM-2 block is zeroed?

The question behind D-dfd-orch-40. A docking test that stubs the 650 MB encoder with zeros
drives the score network to ~1e17 and then to NaN, and the port was suspected. This runs the
same experiment on upstream's own ``TensorProductScoreModel``, unmodified, in the reference
environment: one complex, pose 0, step 0, with ``data['receptor'].x[:, 1:]`` replaced by the
constant the stub uses. If upstream's scores are the same size as ours, a zeroed embedding is
out-of-distribution input rather than a port defect.

    cd ~/scratch/dfd/upstream && PYTHONPATH=. ~/scratch/dfd/venv-ref/bin/python \\
      /home/moritz/.coworker/wt/dfd-parity/scripts/diffdock_port/zero_esm_upstream.py \\
      --id 7A9E_R4W --fill 0,0.02,real --out /tmp/dfd_zeroesm.json
"""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="7A9E_R4W")
    ap.add_argument("--fill", default="0,0.02,real",
                    help="comma list: a constant for every ESM column, or 'real' for the "
                         "embedding upstream computed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fills = a.fill.split(",")

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

    results = {}
    for fill in fills:
        state = {"step": -1, "rows": []}
        fwd = model.forward

        def forward(data, _fill=fill, _state=state):
            _state["step"] += 1
            if _fill != "real":
                data["receptor"].x = data["receptor"].x.clone()
                data["receptor"].x[:, 1:] = float(_fill)
            out = fwd(data)
            if _state["step"] == 0:
                g = lambda t: t.detach()
                _state["rows"] = {
                    "tr_absmax": float(g(out[0]).abs().max()),
                    "rot_absmax": float(g(out[1]).abs().max()),
                    "tor_absmax": float(g(out[2]).abs().max()),
                    "tr_nan": int(torch.isnan(g(out[0])).sum()),
                    "rot_nan": int(torch.isnan(g(out[1])).sum()),
                    "esm_std": float(g(data["receptor"].x)[:, 1:].std()),
                }
            return out

        model.forward = forward
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        dl = [copy.deepcopy(graph) for _ in range(n)]
        randomize_position(dl, sma.no_torsion, False, sma.tr_sigma_max,
                           initial_noise_std_proportion=inf["initial_noise_std_proportion"])
        try:
            sampling(data_list=dl, model=model, inference_steps=1,
                     tr_schedule=sched[:2], rot_schedule=sched[:2], tor_schedule=sched[:2],
                     device=device, t_to_sigma=t_to_sigma, model_args=sma,
                     confidence_model=None, confidence_data_list=None,
                     confidence_model_args=None, batch_size=n,
                     no_final_step_noise=inf["no_final_step_noise"],
                     temp_sampling=[inf["temp_sampling_tr"], inf["temp_sampling_rot"],
                                    inf["temp_sampling_tor"]],
                     temp_psi=[inf["temp_psi_tr"], inf["temp_psi_rot"], inf["temp_psi_tor"]],
                     temp_sigma_data=[inf["temp_sigma_data_tr"], inf["temp_sigma_data_rot"],
                                      inf["temp_sigma_data_tor"]])
        except Exception as exc:                      # upstream itself can fail on the junk
            state["rows"] = dict(state["rows"] or {}, raised=f"{type(exc).__name__}: {exc}")
        model.forward = fwd
        results[fill] = state["rows"]
        print(fill, state["rows"], flush=True)

    out = {"complex": cid, "seed": a.seed, "step": 0, "upstream": True, "fills": results}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
