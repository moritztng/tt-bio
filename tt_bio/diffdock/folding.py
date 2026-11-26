"""Fold a sequence-only receptor before docking it.

Upstream imports fair-esm's `esmfold_v1` and runs it on a CUDA GPU
(`utils/inference_utils.py:166-176`). tt-bio folds with its own esmfold2 on the
card instead (D-dfd-input-5): same contract, one structure file per complex written
under the output directory and reused if it is already there, but a better model
and no second folding stack to maintain.

This module is the bridge only. It holds no device code: the fold goes through
`tt-bio predict --model esmfold2`, the same path a user gets, so multi-card
fan-out, the job controller and the weight cache all apply unchanged.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

#: The suffix tt-bio's predict writes. DiffDock's receptor parser reads it.
STRUCTURE_SUFFIX = ".cif"


def write_fasta(sequence: str, path: Path, *, name: str = "receptor") -> Path:
    """Write the one-chain-per-record FASTA tt-bio's predict takes.

    DiffDock separates receptor chains with ':' (upstream's own convention,
    `utils/inference_utils.py:148`); tt-bio's FASTA gives each chain a record.
    """
    chains = [c for c in sequence.split(":") if c]
    if not chains:
        raise ValueError("receptor sequence is empty")
    records = []
    for i, chain in enumerate(chains):
        records.append(f">{chr(ord('A') + i)}|protein\n{chain}")
    path.write_text("\n".join(records) + "\n")
    return path


def fold_with_esmfold2(
    sequence: str,
    out_path: str | Path,
    *,
    device_ids: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> Path:
    """Fold `sequence` with tt-bio's esmfold2 and leave the structure at `out_path`.

    Raises `RuntimeError` if the fold produced no structure, so the complex is
    isolated as a failure rather than docked against nothing.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ttbio-dfd-fold-") as tmp:
        tmp = Path(tmp)
        fasta = write_fasta(sequence, tmp / "receptor.fasta")
        cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(fasta),
               "--model", "esmfold2", "--out_dir", str(tmp / "out")]
        if device_ids:
            cmd += ["--device_ids", device_ids]
        cmd += list(extra_args)
        subprocess.run(cmd, check=True)
        produced = sorted((tmp / "out").rglob(f"*{STRUCTURE_SUFFIX}"))
        if not produced:
            raise RuntimeError(f"esmfold2 wrote no structure for {out_path.name}")
        out_path.write_bytes(produced[0].read_bytes())
    return out_path
