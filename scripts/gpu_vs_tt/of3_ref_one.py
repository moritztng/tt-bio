#!/usr/bin/env python3
"""One OpenFold3 GPU reference fold, at one size and one seed, harvested to a CIF.

The reference structure for the openfold3-to-4x accuracy question: the shipped `on` arm and
the fused-SDPA `P` arm sit 5.293 A (512 aa) / 16.321 A (298 aa) apart, and neither can be
called the right basin without the official implementation's own answer on the same input.

This is `gpu5_bench.run_of3` verbatim -- same query schema, same runner YAML, same argv, same
asserts -- with two changes and nothing else:

  * the seed is a parameter instead of the module constant, so a seed ensemble is reachable;
  * `--repeat 0` folds once, because this run measures a structure, not a time.

One process per (size, seed). Re-entering openfold3's click group inside a live process is
untested; a fresh process per fold costs one weight load (~30 s) and removes the question.

    /root/venv-of3/bin/python3 of3_ref_one.py --size 298 --seed 0 --outdir /root/results/of3ref
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1] / "perf" / "size512" / "fixtures"
SEQ = {298: HERE / "fixtures" / "prot300.seq", 512: HERE / "fixtures" / "prot512.seq"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True, choices=[298, 512])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("/root/results/of3ref"))
    ap.add_argument("--checkpoint", default="/root/ckpt/of3-p2-155k.pt")
    ap.add_argument("--work", default="/root/work")
    a = ap.parse_args()

    a3m = FIX / f"cdk2x2_{a.size}.a3m"
    seq_file = SEQ[a.size]
    # The three inputs the TT arms consumed have to be the same bytes here, and that is
    # checked rather than assumed: the .seq the reference folds, the fixture YAML the TT arm
    # folds, and the a3m's own query row are one sequence or this run is not comparable.
    seq = seq_file.read_text().strip()
    yaml_seq = next(l.split("sequence:")[1].strip()
                    for l in (FIX / f"cdk2x2_{a.size}.yaml").read_text().splitlines()
                    if "sequence:" in l)
    a3m_rows = a3m.read_text().split("\n")
    assert len(seq) == a.size, f"{seq_file} is {len(seq)} aa, not {a.size}"
    assert seq == yaml_seq, "the .seq and the fixture YAML disagree"
    assert seq == a3m_rows[1], "the a3m query row is not the target sequence"

    sys.path.insert(0, str(HERE))
    import gpu5_bench  # noqa: E402  -- must import before openfold3 (cueq counters)

    gpu5_bench.SEED = a.seed
    args = types.SimpleNamespace(
        seq_file=seq_file, a3m=a3m, work=a.work, repeat=0,
        checkpoint=a.checkpoint, extra=None,
    )
    res = gpu5_bench.run_of3(args)          # asserts a structure exists and the seed held

    preds = [Path(p) for p in res["predictions"]]
    assert len(preds) == 1, f"expected one structure, got {len(preds)}: {preds}"
    a.outdir.mkdir(parents=True, exist_ok=True)
    cif = a.outdir / f"ref_{a.size}_seed{a.seed}.cif"
    shutil.copyfile(preds[0], cif)

    import torch
    from importlib.metadata import version
    meta = {
        "size": a.size, "seed": a.seed, "cif": str(cif),
        "recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1,
        "templates": False, "msa_rows": res["msa_rows"], "n_residues": res["n_residues"],
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "packages": {p: version(p) for p in ("torch", "openfold3", "cuequivariance_torch")},
        "checkpoint_sha256_head": subprocess.run(
            ["sha256sum", a.checkpoint], capture_output=True, text=True,
        ).stdout.split()[0][:16],
        "a3m_sha256_head": subprocess.run(
            ["sha256sum", str(a3m)], capture_output=True, text=True,
        ).stdout.split()[0][:16],
        "kernel_counts_total": res["kernel_counts_total"],
        "cli_argv": res["cli_argv"],
        "cold_s": res.get("cold_s"),
        "host": os.uname().nodename,
    }
    (a.outdir / f"ref_{a.size}_seed{a.seed}.json").write_text(json.dumps(meta, indent=1))
    tri = meta["kernel_counts_total"].get("triangle.triangle_attention", 0)
    print(f"wrote {cif}  cold {meta['cold_s']} s  triangle_attention calls {tri}", flush=True)
    assert tri > 0, ("cuEquivariance triangle attention never ran: this fold is not the "
                     "reference's shipped GPU path")


if __name__ == "__main__":
    main()
