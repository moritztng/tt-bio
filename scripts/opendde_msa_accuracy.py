"""Controlled OpenDDE accuracy read on the complete 7ROA construct.

Uses the query from examples/msa/seq2.a3m so the MSA and no-MSA runs have the
same 136-residue input. Scores only residues resolved in the 7ROA structure.

Run on qb2 card 0:
  TT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/opendde_msa_accuracy.py

Set OPENDDE_USE_MSA=0 for the length-matched no-MSA control. Production defaults
are 10 recycling cycles, 200 diffusion steps, and five samples (seeds 0..4).
"""
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import numpy as np
import torch
import ttnn

from tt_bio.data import const
from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
from tt_bio.protenix_data import build_complex_features
from tt_bio.tenstorrent import get_device

torch.set_grad_enabled(False)

ROOT = Path(__file__).resolve().parent.parent
A3M_PATH = ROOT / "examples" / "msa" / "seq2.a3m"
GT_PATH = ROOT / "examples" / "ground_truth_structures" / "prot.cif"
_LETTER_TO_RES = {v: k for k, v in const.prot_token_to_letter.items()}


def _a3m_query(a3m: str) -> str:
    return next(line.strip() for line in a3m.splitlines()
                if line.strip() and not line.startswith(">"))


def _predicted_ca(coords: np.ndarray, sequence: str) -> np.ndarray:
    indices = []
    offset = 0
    for i, letter in enumerate(sequence):
        residue = _LETTER_TO_RES[letter]
        atoms = list(const.ref_atoms[residue])
        if i == len(sequence) - 1:
            atoms.append("OXT")
        indices.append(offset + atoms.index("CA"))
        offset += len(atoms)
    return coords[np.asarray(indices)]


def _score_samples(coords: torch.Tensor, sequence: str) -> list[dict[str, float]]:
    spec = importlib.util.spec_from_file_location(
        "tt_bio_test_structure", ROOT / "tests" / "test_structure.py")
    structure_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(structure_test)
    (chain_id, truth_by_pos), = structure_test.get_ca_atoms(str(GT_PATH)).items()
    positions = sorted(truth_by_pos)
    truth = np.asarray([truth_by_pos[p] for p in positions])

    results = []
    for sample in coords.numpy():
        pred = _predicted_ca(sample, sequence)
        pred = pred[np.asarray(positions) - 1]
        dev = structure_test._kabsch_deviations(pred, truth)
        results.append({
            "ca_rmsd": float(np.sqrt((dev ** 2).mean())),
            "tm": float(structure_test._tm_score(dev, len(positions))),
        })
    print(f"scored {len(positions)} resolved residues from chain {chain_id}")
    return results


def _confidence_score(conf: dict) -> float:
    ptm, iptm = float(conf.get("ptm", 0.0)), float(conf.get("iptm", 0.0))
    if iptm > 0.0:
        return 0.8 * iptm + 0.2 * ptm
    return ptm if ptm > 0.0 else float(conf["plddt"])


def main() -> None:
    use_msa = os.environ.get("OPENDDE_USE_MSA", "1") != "0"
    n_cycles = int(os.environ.get("OPENDDE_NCYCLES", "10"))
    n_step = int(os.environ.get("OPENDDE_NSTEP", "200"))
    n_sample = int(os.environ.get("OPENDDE_NSAMPLE", "5"))
    seed = int(os.environ.get("OPENDDE_SEED", "0"))

    a3m = A3M_PATH.read_text()
    sequence = _a3m_query(a3m).replace("X", "M")
    feats = build_complex_features([(sequence, a3m if use_msa else None, "protein")])
    print(f"input: N_res={len(sequence)} N_msa={feats['msa'].shape[0]} "
          f"cycles={n_cycles} steps={n_step} samples={n_sample}", flush=True)

    started = time.time()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = OpenDDE(load_opendde_checkpoint(), ckc, dev)
    print(f"[{time.time() - started:.1f}s] model built", flush=True)
    coords, confidence = model.fold(
        feats, n_step=n_step, n_cycles=n_cycles, seed=seed,
        n_sample=n_sample, return_confidence=True)
    confidence = confidence if isinstance(confidence, list) else [confidence]
    accuracy = _score_samples(coords, sequence)

    rows = []
    for i, (acc, conf) in enumerate(zip(accuracy, confidence)):
        rows.append({
            "sample": i, "seed": seed + i, **acc,
            "plddt": float(conf["plddt"]),
            "ptm": float(conf.get("ptm", 0.0)),
            "confidence_score": _confidence_score(conf),
        })
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    selected = max(range(len(rows)), key=lambda i: rows[i]["confidence_score"])
    oracle = min(range(len(rows)), key=lambda i: rows[i]["ca_rmsd"])
    result = {
        "use_msa": use_msa, "n_res": len(sequence),
        "n_msa": int(feats["msa"].shape[0]), "n_cycles": n_cycles,
        "n_step": n_step, "rows": rows,
        "confidence_selected": rows[selected], "oracle": rows[oracle],
        "mean_ca_rmsd": float(np.mean([row["ca_rmsd"] for row in rows])),
        "elapsed_s": time.time() - started,
    }
    suffix = "msa" if use_msa else "nomsa"
    output = Path(f"/tmp/opendde_7roa_full_{suffix}.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(coords, f"/tmp/opendde_7roa_full_{suffix}.pt")
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
