"""One BoltzGen design arm: seed the host, run the design step, digest what it wrote.

The two arms differ in exactly one environment variable, `TT_BIO_ATOM_L1`, which sets a
`ttnn.MemoryConfig` inside the atom branch of `AttentionPairBias` and nothing else. BoltzGen
reaches that branch through `TTScoreModelAdapter`, which subclasses the same
`tenstorrent.DiffusionModule` Boltz-2 folds through, so the flag lands on a design job the
moment it becomes a default.

BoltzGen's CLI has no seed: the sampler draws its noise from torch's global RNG and the
pipeline never touches it, so two runs of the shipped command produce different structures and
no digest compare is possible. Seeding here, before anything imports, makes the host draws
identical between arms. Whatever is left after that is the device, which is the thing under
test.

Writes `arm.json` beside the designs: the arm, the per-arm `ATOM_L1_STATS` census, the
sampling protocol read back off the config the run actually used, and a digest per produced
structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path


def _digests(cif: Path) -> dict:
    """File digest plus a coordinate-only digest, so metadata cannot mask a moved atom."""
    raw = cif.read_bytes()
    out = {"file_sha256": hashlib.sha256(raw).hexdigest()}
    try:
        import gemmi

        st = gemmi.read_structure(str(cif))
        h = hashlib.sha256()
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        h.update(
                            f"{chain.name}|{res.seqid.num}|{res.name}|{atom.name}|"
                            f"{atom.pos.x!r}|{atom.pos.y!r}|{atom.pos.z!r}\n".encode()
                        )
        out["coord_sha256"] = h.hexdigest()
    except Exception as exc:  # noqa: BLE001 -- a parse failure must not hide the file digest
        out["coord_sha256"] = f"unavailable: {exc}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arm", required=True, choices=("base", "l1"))
    ap.add_argument("--device", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_designs", type=int, default=1)
    args = ap.parse_args()

    # Set before ttnn or tt_bio.tenstorrent import: the gate reads the flag at module scope.
    os.environ["TT_BIO_ATOM_L1"] = "1" if args.arm == "l1" else "0"
    os.environ.setdefault("TT_BIO_ATOM_L1_TRACE", "1")

    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    argv = [
        "run", str(args.spec),
        "--output", str(args.out),
        "--num_designs", str(args.num_designs),
        "--device_ids", str(args.device),
        "--steps", "design",
        "--debug", "--log",
    ]

    from tt_bio.main import _run_boltzgen_cli

    t0 = time.time()
    _run_boltzgen_cli("tt-bio design", argv)
    wall = time.time() - t0

    from tt_bio import tenstorrent

    produced = sorted((args.out / "intermediate_designs").glob("*.cif"))
    record = {
        "arm": args.arm,
        "atom_l1_env": os.environ["TT_BIO_ATOM_L1"],
        "seed": args.seed,
        "device_ids": args.device,
        "wall_s": round(wall, 3),
        "atom_l1_stats": dict(tenstorrent.ATOM_L1_STATS),
        "designs": {p.name: _digests(p) for p in produced},
    }
    # The protocol the run actually used, read back off the config it wrote, not off the
    # command line: a design step that silently downshifted its sampling would otherwise
    # read as a clean pass.
    cfg = args.out / "config.yaml"
    if cfg.exists():
        text = cfg.read_text()
        record["config_sampling_steps"] = [
            ln.strip() for ln in text.splitlines()
            if "sampling_steps" in ln or "recycling_steps" in ln
        ]
    (args.out / "arm.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if produced else 1


if __name__ == "__main__":
    sys.exit(main())
