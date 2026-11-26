"""Capture upstream goldens for the DiffDock-L confidence model.

    scripts/diffdock_port/confidence_goldens.sh --id 8G43_ZU6

`dfd-ref`'s `capture_goldens.py` attaches a `GoldenStore` to both networks, but its
`max_calls` budget is spent by the score model's 19 steps before the confidence pass runs
once at the end, so `state/dfd/goldens/*__mod/` holds 298 `score__*` files and no `conf__*`
ones. The per-pose `[N, 2]` output survived in `poses.npz`; the per-module activations did
not.

This recaptures the confidence half alone, and **does not re-run the rollout**. The
confidence model is a pure function of the receptor graph and the final ligand coordinates,
and those coordinates are already in `poses.npz`. Upstream's own call site
(`utils/sampling.py:208-227`) does exactly three things before the forward pass: copy the
sampled `['ligand'].pos` into a separate confidence graph, `set_time(..., 0, 0, 0, 0, ...)`,
and `nan_to_num(nan=-1000)` the result. Reproducing those three on the captured poses gives
the identical forward pass for a fraction of the cost.

That reconstruction is not taken on trust. `--check` recomputes the `[N, 2]` tensor and
compares it against the `confidence_raw` the full-rollout capture already wrote; if the two
disagree the inputs were rebuilt wrongly and the script fails rather than writing goldens
that grade the port against the wrong thing.

Two upstream details that are easy to lose and that the rebuild depends on:

* **`crop_beyond` is not applied here.** The score checkpoint sets it (D-dfd-orch-11); the
  confidence checkpoint has no such key, and `sampling.py:213` guards on `hasattr`. Cropping
  the confidence graph would change its receptor and every number below.
* **The poses in `poses.npz` carry `original_center` added back**, because that is what a
  user receives. The network sees centred coordinates, so the centre is subtracted again
  here. Feeding the uncentred poses produces finite, plausible, wrong confidences.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

D = Path.home() / "scratch/dfd"
GOLD = Path.home() / ".coworker/state/dfd/goldens"
PB = D / "data/posebusters_dl/sel"

#: The complexes whose `poses.npz` carries a `confidence_raw` tensor to rebuild against.
DEFAULT_IDS = ("8G43_ZU6", "8AEU_M0L", "7A9E_R4W")


def _upstream_imports():
    """Import the upstream package from its own checkout, with its CWD conventions.

    `utils/so3.py` and `utils/torus.py` build and cache `.npy` lookup tables in the current
    directory at import time, so the working directory has to be the upstream root before
    the first import, not just on sys.path.
    """
    sys.path.insert(0, str(D / "upstream"))
    sys.path.insert(0, str(D / "scripts"))


def build_confidence_graph(cid: str, cma: Namespace, lm):
    """One uncollated confidence `Data` for `cid`, at the confidence checkpoint's settings.

    `use_original_model_cache: false` in the confidence `model_parameters.yml` is why this
    is a second dataset build rather than a reuse of the score model's graph: the two
    checkpoints disagree on `receptor_radius` (15 vs 30) and on `all_atoms`, so they see
    different receptor graphs for the same protein.
    """
    from utils.inference_utils import InferenceDataset

    prot, lig = PB / cid / f"{cid}_protein.pdb", PB / cid / f"{cid}_ligand.sdf"
    out_dir = D / "runs/confidence_goldens" / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = InferenceDataset(
        out_dir=str(out_dir), complex_names=[cid], protein_files=[str(prot)],
        ligand_descriptions=[str(lig)], protein_sequences=[None], lm_embeddings=True,
        receptor_radius=cma.receptor_radius, remove_hs=cma.remove_hs,
        c_alpha_max_neighbors=cma.c_alpha_max_neighbors, all_atoms=cma.all_atoms,
        atom_radius=cma.atom_radius, atom_max_neighbors=cma.atom_max_neighbors,
        precomputed_lm_embeddings=lm, knn_only_graph=False)
    return ds[0]


def run_confidence(cid: str, store=None, *, ids_dir: Path | None = None):
    """Upstream's confidence forward pass on the captured final poses. Returns `[N, 2]`."""
    import copy

    from torch_geometric.loader import DataLoader
    from utils.utils import get_model
    from utils.sampling import set_time
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
    from capture_goldens import lm_embeddings_for

    gold_dir = (ids_dir or GOLD) / f"{cid}__s0"
    poses = np.load(gold_dir / "poses.npz")
    pos, center = poses["pos"], poses["center"]
    n = len(pos)

    inf = yaml.safe_load((D / "upstream/default_inference_args.yaml").read_text())
    with open(D / "ckpt/extracted/confidence_model/model_parameters.yml") as f:
        cma = Namespace(**yaml.full_load(f))
    with open(D / "ckpt/extracted/score_model/model_parameters.yml") as f:
        sma = Namespace(**yaml.full_load(f))

    prot = PB / cid / f"{cid}_protein.pdb"
    lm = lm_embeddings_for(cid, str(prot), str(D / "runs/confidence_goldens" / cid))

    device = torch.device("cpu")
    cmodel = get_model(cma, device, t_to_sigma=partial(t_to_sigma_compl, args=sma),
                       no_parallel=True, confidence_mode=True,
                       old=inf.get("old_filtering_model", True))
    cmodel.load_state_dict(
        torch.load(D / "ckpt/extracted/confidence_model" / inf["confidence_ckpt"],
                   map_location="cpu", weights_only=False), strict=True)
    cmodel.eval()

    graph = build_confidence_graph(cid, cma, lm)
    batch = next(iter(DataLoader([copy.deepcopy(graph) for _ in range(n)], batch_size=n)))
    # The network sees centred coordinates; poses.npz stores them with original_center
    # added back, which is what a caller receives.
    batch["ligand"].pos = torch.from_numpy(pos - center).float().reshape(-1, 3)
    set_time(batch, 0, 0, 0, 0, n, cma.all_atoms, device)

    if store is not None:
        store.attach(cmodel, prefix="conf.")
    with torch.no_grad():
        out = cmodel(batch)
    return torch.nan_to_num(out, nan=-1000).cpu().numpy(), graph


#: The confidence graph tensors the port's `ConfidenceModel.forward` takes, in its argument
#: order. Dumping them lets the parity grade feed BOTH sides the identical graph, so a
#: disagreement is the network's and not the dataset builder's. Graph construction is
#: `dfd-input`'s to grade, and conflating the two would hide either.
INPUT_KEYS = ("lig_x", "bond_index", "bond_attr", "rec_x", "rec_pos", "rec_edge_index",
              "atom_x", "atom_pos", "atom_edge_index", "atom_rec_edge_index")


def dump_inputs(cid: str) -> dict:
    """Write upstream's confidence graph for `cid` as plain arrays.

    `rec_x` carries the ESM-2 block already concatenated, exactly as the dataset hands it to
    `rec_node_embedding`; the sigma columns are NOT included, because the model appends those
    itself from the diffusion time.
    """
    from argparse import Namespace as NS

    from capture_goldens import lm_embeddings_for

    with open(D / "ckpt/extracted/confidence_model/model_parameters.yml") as f:
        cma = NS(**yaml.full_load(f))
    prot = PB / cid / f"{cid}_protein.pdb"
    lm = lm_embeddings_for(cid, str(prot), str(D / "runs/confidence_goldens" / cid))
    g = build_confidence_graph(cid, cma, lm)

    arrays = {
        "lig_x": g["ligand"].x, "bond_index": g["ligand", "ligand"].edge_index,
        "bond_attr": g["ligand", "ligand"].edge_attr,
        "rec_x": g["receptor"].x, "rec_pos": g["receptor"].pos,
        "rec_edge_index": g["receptor", "receptor"].edge_index,
        "atom_x": g["atom"].x, "atom_pos": g["atom"].pos,
        "atom_edge_index": g["atom", "atom"].edge_index,
        "atom_rec_edge_index": g["atom", "receptor"].edge_index,
    }
    out = {k: np.asarray(v.cpu()) for k, v in arrays.items()}
    path = GOLD / f"{cid}__s0" / "confidence_inputs.npz"
    np.savez_compressed(path, **out)
    print(f"{cid}: inputs -> {path.name} "
          + " ".join(f"{k}{tuple(v.shape)}" for k, v in out.items()), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", dest="ids", default=None)
    ap.add_argument("--no-goldens", action="store_true",
                    help="only recheck the rebuild against poses.npz")
    ap.add_argument("--max-calls", type=int, default=1)
    ap.add_argument("--inputs-only", action="store_true",
                    help="dump the confidence graph tensors and stop")
    args = ap.parse_args()
    ids = args.ids or list(DEFAULT_IDS)

    _upstream_imports()
    import dfd_ref_lib as L

    torch.set_grad_enabled(False)
    if args.inputs_only:
        for cid in ids:
            dump_inputs(cid)
        return 0

    report = {}
    for cid in ids:
        dump_inputs(cid)
        gold_dir = GOLD / f"{cid}__s0"
        store = None
        if not args.no_goldens:
            store = L.GoldenStore(GOLD / f"{cid}__mod", cid, 0, max_calls=args.max_calls)
        out, _ = run_confidence(cid, store)
        if store is not None:
            rows: list = []
            store.save(rows)
            (GOLD / f"{cid}__mod" / f"{cid}__conf__index.json").write_text(
                json.dumps(rows, indent=1, default=str))
            print(f"{cid}: {len(rows)} confidence module goldens", flush=True)

        ref = np.load(gold_dir / "poses.npz")["confidence_raw"]
        if ref.size == 0:
            raise SystemExit(f"{cid}: poses.npz has no confidence_raw to check against")
        if out.shape != ref.shape:
            raise SystemExit(f"{cid}: rebuilt {out.shape}, capture has {ref.shape}")
        amax = float(np.max(np.abs(out - ref)))
        rel = float(np.max(np.abs(out - ref) / np.maximum(np.abs(ref), 1e-6)))
        rank_same = bool(np.array_equal(np.argsort(-out[:, 0], kind="stable"),
                                        np.argsort(-ref[:, 0], kind="stable")))
        np.savez_compressed(gold_dir / "confidence.npz", confidence_raw=out,
                            rebuilt_max_abs_err=np.float64(amax))
        report[cid] = {"max_abs_err": amax, "max_rel_err": rel, "rank_order_same": rank_same,
                       "n_poses": int(out.shape[0])}
        print(f"{cid}: max|d| {amax:.3e}  rel {rel:.3e}  rank order same: {rank_same}",
              flush=True)

    (GOLD / "CONFIDENCE-REBUILD.json").write_text(json.dumps(report, indent=1))
    worst = max(r["max_abs_err"] for r in report.values())
    # The rebuild replays the identical forward pass, so it should agree to float32
    # rounding on a reduction over the whole ligand, not merely correlate.
    if worst > 1e-4:
        print(f"REBUILD MISMATCH: worst max|d| {worst:.3e} > 1e-4", file=sys.stderr)
        return 1
    print(f"rebuild verified on {len(report)} complexes, worst max|d| {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
