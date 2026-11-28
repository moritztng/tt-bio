"""Dump the score network's own inputs at the step where its output reaches 1e15.

Four PoseBusters complexes (7AVI_S2Q, 7TBU_S3P, 7X5N_5M5, 7ZCC_OGA) die in the Kabsch SVD
because the tr/rot heads return a *finite* ~1e15 that no NaN guard keys on (dfd-eval,
`state/dfd/note-dfd-eval-to-dfd-parity-nan.md`).  Upstream docks all four with its heads
staying O(10).  Whether that is a parity break in this network or a pathological pose the
rollout wandered into can only be settled on ONE input, so this records that input.

Docks one complex exactly as the 428-set grade does (host torch, seed 42, the default
recipe) with `ScoreModel.forward` wrapped.  Every call's inputs are kept in a rolling
buffer; the first call whose output exceeds `--threshold` is written out together with the
*same pose's* call one step earlier, in the npz contract `zero_esm_upstream.py --dump`
writes, so either side can be replayed through upstream or through the port.

    PYTHONPATH=$PWD python3 scripts/diffdock_port/blowup_dump.py 7ZCC_OGA \\
        --out ~/scratch/dfd/parity-blowup
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PB = Path.home() / "scratch/dfd/pb/posebusters_benchmark_set"


def smiles_of(cid: str):
    from rdkit import Chem
    m = next(Chem.SDMolSupplier(f"{PB}/{cid}/{cid}_ligand.sdf", sanitize=True, removeHs=True),
             None)
    return None if m is None else Chem.MolToSmiles(Chem.RemoveHs(m))


def census(t):
    if t is None:
        return {"n": 0}
    f = t[torch.isfinite(t)]
    return {"n": int(t.numel()), "nan": int(torch.isnan(t).sum()),
            "inf": int(torch.isinf(t).sum()),
            "absmax": float(f.abs().max()) if f.numel() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cid")
    ap.add_argument("--out", default=str(Path.home() / "scratch/dfd/parity-blowup"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=1e5)
    ap.add_argument("--samples", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out) / a.cid
    out.mkdir(parents=True, exist_ok=True)

    from tt_bio.diffdock import run, score_model
    from tt_bio.diffdock.job import DockJob, DockOptions

    fwd = score_model.ScoreModel.forward
    # one entry per forward call: (call_index, the twelve inputs, the three outputs)
    history: list[dict] = []
    state = {"calls": 0, "dumped": False, "rows": []}

    def wrapped(self, lig_x, lig_pos, bond_index, bond_attr, edge_mask, rec_x, rec_pos,
                rec_edge_index, t_tr, tr_sigma, rot_sigma, tor_sigma, receptor_cache=None):
        o = fwd(self, lig_x, lig_pos, bond_index, bond_attr, edge_mask, rec_x, rec_pos,
                rec_edge_index, t_tr, tr_sigma, rot_sigma, tor_sigma,
                receptor_cache=receptor_cache)
        i = state["calls"]
        state["calls"] += 1
        tr, rot, tor = o[0], o[1], o[2]
        rec = {
            "call": i,
            "t_tr": float(t_tr),
            "tr": census(tr), "rot": census(rot), "tor": census(tor),
            "n_lig": int(lig_pos.shape[0]),
            "n_rec": int(rec_pos.shape[0]),
            "lig_absmax": float(lig_pos.abs().max()),
            "lig_spread": float(lig_pos.std(0).norm()),
            "centre_dist": float((lig_pos.mean(0) - rec_pos.mean(0)).norm()),
        }
        state["rows"].append(rec)
        snap = {
            "call": i,
            "lig_x": lig_x.numpy(), "lig_pos": lig_pos.detach().numpy(),
            "bond_index": bond_index.numpy().astype(np.int32),
            "bond_attr": bond_attr.numpy(), "edge_mask": edge_mask.numpy().astype(np.bool_),
            "rec_x": rec_x.numpy(), "rec_pos": rec_pos.detach().numpy(),
            "rec_edge_index": rec_edge_index.numpy().astype(np.int32),
            "t_tr": np.asarray(float(t_tr)), "t_rot": np.asarray(float(t_tr)),
            "t_tor": np.asarray(float(t_tr)),
            "tr_sigma": np.asarray(float(tr_sigma)),
            "rot_sigma": np.asarray(float(rot_sigma)),
            "tor_sigma": np.asarray(float(tor_sigma)),
            "tr": tr.detach().numpy().astype(np.float64),
            "rot": rot.detach().numpy().astype(np.float64),
            "tor": tor.detach().numpy().astype(np.float64),
        }
        history.append(snap)
        if len(history) > 64:
            history.pop(0)

        biggest = max(float(x.abs().max()) if x is not None and x.numel() else 0.0
                      for x in (tr, rot, tor) if x is not None)
        if not state["dumped"] and np.isfinite(biggest) and biggest > a.threshold:
            state["dumped"] = True
            np.savez_compressed(out / "blowup.npz", **snap)
            # the same pose one step earlier: the rollout advances the poses in a fixed
            # order, so the call `--samples` back is the same sample at the previous step.
            n = a.samples or state["samples"]
            prev = next((h for h in history if h["call"] == i - n), None)
            if prev is not None:
                np.savez_compressed(out / "prev.npz", **prev)
            print(f"DUMPED at call {i}: absmax {biggest:.4g}", flush=True)
        return o

    score_model.ScoreModel.forward = wrapped

    opts = DockOptions(seed=a.seed)
    state["samples"] = opts.diffusion_samples
    job = DockJob(name=a.cid, receptor_path=PB / a.cid / f"{a.cid}_protein.pdb",
                  ligand=smiles_of(a.cid))
    try:
        summary = run.dock([job], opts, out, isolate=True)
        err = summary.get("failed") or None
    except Exception as exc:                       # the SVD raises for the whole batch
        err = f"{type(exc).__name__}: {exc}"
    print("SUMMARY", err or "docked", flush=True)
    (out / "census.json").write_text(json.dumps(
        {"complex": a.cid, "seed": a.seed, "samples": state["samples"],
         "error": err, "calls": state["rows"]}, indent=1))
    print("wrote", out / "census.json", flush=True)


if __name__ == "__main__":
    main()
