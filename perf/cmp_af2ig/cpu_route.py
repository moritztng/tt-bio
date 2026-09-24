"""The af2ig route end to end on host torch: real parameters, real numbers, no card.

What this answers is about the WIRING, not about AlphaFold2: does a submission reach the model,
does the binder sequence actually change the answer, and does the same submission twice give the
same answer. The trunk's device parity is a different question and a committed one
(tests/test_af2_device_floor.py against docs/implementation-parity-data/af2ig-trunk-device.json).

    TT_VISIBLE_DEVICES= PYTHONPATH=. env/bin/python3 perf/cmp_af2ig/cpu_route.py \
        --params ~/.boltz/af2/params/params_model_1_ptm.npz --out perf/cmp_af2ig/cpu_route.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio import af2ig  # noqa: E402
from tt_bio.af2_reference import load_af2_model  # noqa: E402
from tt_bio.af2_weights import load_af2_state_dict  # noqa: E402

EXAMPLE = REPO / "examples" / "af2_designed_complex.yaml"
#: The example's binder, and the same 16 residues in a different order. Same composition, same
#: backbone, same everything the model reads except the sequence: if the numbers do not move,
#: the sequence is not reaching the trunk.
BINDER = "LYRWIKSVDPSRPVQY"
SCRAMBLED = "VQYSVDPSRPLYRWIK"


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a.astype(np.float32)).tobytes()).hexdigest()[:16]


def run(model, spec, recycles: int) -> dict:
    t0 = time.perf_counter()
    pred = af2ig.fold(model, spec, recycles=recycles)
    return {"tokens": pred.tokens, "binder_residues": pred.binder_length,
            "atoms": int(pred.atom_array.array_length()),
            "fold_s": round(time.perf_counter() - t0, 3),
            "coords_sha16": _digest(pred.coords), "metrics": pred.metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="~/.boltz/af2/params/params_model_1_ptm.npz")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_af2_model(load_af2_state_dict(str(Path(args.params).expanduser())),
                           template=True).eval()
    load_s = round(time.perf_counter() - t0, 3)

    spec = af2ig.read_af2ig_input(EXAMPLE)
    assert spec.binder_sequence == BINDER
    other = af2ig.AF2IGInput(structure=spec.structure, binder_sequence=SCRAMBLED,
                             target_chain=spec.target_chain, binder_chain=spec.binder_chain)
    legs = {"example": run(model, spec, args.recycles),
            "example_again": run(model, spec, args.recycles),
            "scrambled_binder": run(model, other, args.recycles)}
    a, b, c = legs["example"], legs["example_again"], legs["scrambled_binder"]
    report = {
        "host": socket.gethostname(), "arm": "host torch (no ttnn, no device)",
        "torch": torch.__version__, "python": platform.python_version(),
        "params": str(Path(args.params).expanduser()),
        "params_sha256_16": hashlib.sha256(
            Path(args.params).expanduser().read_bytes()).hexdigest()[:16],
        "recycles": args.recycles, "passes_per_fold": args.recycles + 1,
        "model_load_s": load_s, "legs": legs,
        "deterministic": a["coords_sha16"] == b["coords_sha16"] and a["metrics"] == b["metrics"],
        "sequence_reaches_the_model": a["coords_sha16"] != c["coords_sha16"],
        "iptm_delta_on_scramble": round(c["metrics"]["iptm"] - a["metrics"]["iptm"], 4),
    }
    text = json.dumps(report, indent=1) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text)
    return 0 if report["deterministic"] and report["sequence_reaches_the_model"] else 1


if __name__ == "__main__":
    sys.exit(main())
