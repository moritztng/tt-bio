#!/usr/bin/env python3
"""Verify the SHIPPED DEFAULT, not the arm the A/B harness set in-process.

The A/B run switched both levers by assigning module globals. That proves the code paths, not the
defaults a user gets, and the two are different claims (memory
`merged-lever-defaults-off-is-not-a-landed-win`, `verify-the-deployed-artifact-not-your-own-change`).
So this runs the real CLI, in its own process, with NO lever variable in the environment, through
the worker spawn a user's fold goes through, and requires the coordinates to be identical to the
AB arm of the A/B run.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "other512"))
from cif_rmsd import read_atoms                                               # noqa: E402

FIX = REPO / "perf" / "size512" / "fixtures"
AB = REPO / "perf" / "b2x-integrate" / "cif512" / "512_AB_0" / "cdk2x2_512.cif"


def main() -> int:
    for v in ("BOLTZ2_TOKEN_DIT_SDPA", "TT_BIO_ATOM_AXIS_BUCKET"):
        assert v not in os.environ, f"{v} is set; this pass must read the DEFAULT"
    sys.path.insert(0, str(REPO))
    import tt_bio.tenstorrent as TT
    assert TT._B2_TOKEN_DIT_SDPA is True and TT._ATOM_AXIS_BUCKET is True, \
        f"defaults did not flip: sdpa={TT._B2_TOKEN_DIT_SDPA} bucket={TT._ATOM_AXIS_BUCKET}"
    from tt_bio.main import _read_bio_chains

    work = Path(tempfile.mkdtemp(prefix="b2x-verify-"))
    msa = work / "msa"; msa.mkdir(parents=True)
    tgt = FIX / "cdk2x2_512.yaml"
    seq = _read_bio_chains(tgt)[0][1]
    a3m = (FIX / "cdk2x2_512.a3m").read_text()
    assert a3m.split("\n")[1] == seq
    (msa / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(a3m)

    out = work / "out"
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(tgt),
           "--model", "boltz2", "--msa_dir", str(msa), "--out_dir", str(out),
           "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
           "--seed", "0", "--output_format", "cif"]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print("+", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    tail = "\n".join(l for l in r.stdout.splitlines()[-6:])
    print(f"rc={r.returncode} wall={wall:.3f}s\n{tail}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1

    cifs = sorted(out.rglob("*.cif"))
    assert len(cifs) == 1, f"expected one CIF, got {cifs}"
    k_cli, x_cli = read_atoms(cifs[0])
    k_ab, x_ab = read_atoms(AB)
    same_keys = k_cli == k_ab
    same_xyz = bool(np.array_equal(x_cli, x_ab))
    print(f"CLI default fold : {len(x_cli)} atoms  file sha256 "
          f"{hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]}")
    print(f"AB arm of the A/B: {len(x_ab)} atoms  file sha256 "
          f"{hashlib.sha256(AB.read_bytes()).hexdigest()[:16]}")
    print(f"atom identity identical: {same_keys}   coordinates identical: {same_xyz}")
    if not (same_keys and same_xyz):
        d = np.abs(x_cli - x_ab).max() if same_keys else float("nan")
        print(f"MISMATCH, max |delta| = {d}")
        return 1
    print("PASS: the shipped default reproduces the AB arm exactly, end to end through the CLI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
