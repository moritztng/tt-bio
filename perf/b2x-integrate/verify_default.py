#!/usr/bin/env python3
"""Verify the SHIPPED DEFAULT, not the arm the A/B harness set in-process.

The A/B run switched both levers by assigning module globals. That proves the code paths, not the
defaults a user gets, and the two are different claims (memory
`merged-lever-defaults-off-is-not-a-landed-win`, `verify-the-deployed-artifact-not-your-own-change`).
So this runs the real CLI, in its own process, through the worker spawn a user's fold goes
through, and requires the coordinates to match a stated reference.

``--arm default`` is the shipped path: no lever variable in the environment, both module globals
must read True. ``--arm off`` is the documented escape hatch, ``BOLTZ2_TOKEN_DIT_SDPA=0
TT_BIO_ATOM_AXIS_BUCKET=0``, which has to keep working as a real CLI invocation and not only as
an in-process arm -- that is the other half of what a platform-neutrality check owes.

``--ref`` is per platform: coordinates are only comparable against an A/B run taken on the same
card type, so a Wormhole pass points it at its own AB (or base) CIF, not at Blackhole's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
BH_AB = REPO / "perf" / "b2x-integrate" / "cif512" / "512_AB_0" / "cdk2x2_512.cif"
LEVERS = ("BOLTZ2_TOKEN_DIT_SDPA", "TT_BIO_ATOM_AXIS_BUCKET")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("default", "off"), default="default")
    ap.add_argument("--ref", type=Path, default=None,
                    help="CIF the coordinates must reproduce; per card type. "
                         f"--arm default on a p300c: {BH_AB}")
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--out", type=Path, help="JSON result")
    ap.add_argument("--keep", type=Path, help="copy the produced CIF here")
    a = ap.parse_args()

    for v in LEVERS:
        assert v not in os.environ, f"{v} is set in the environment; --arm picks the arm"
    sys.path.insert(0, str(REPO))
    import tt_bio.tenstorrent as TT
    assert TT._B2_TOKEN_DIT_SDPA is True and TT._ATOM_AXIS_BUCKET is True, \
        f"defaults did not flip: sdpa={TT._B2_TOKEN_DIT_SDPA} bucket={TT._ATOM_AXIS_BUCKET}"
    from tt_bio.main import _read_bio_chains

    work = Path(tempfile.mkdtemp(prefix="b2x-verify-"))
    msa = work / "msa"; msa.mkdir(parents=True)
    tgt = FIX / f"{a.fixture}.yaml"
    seq = _read_bio_chains(tgt)[0][1]
    a3m = (FIX / f"{a.fixture}.a3m").read_text()
    assert a3m.split("\n")[1] == seq
    (msa / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(a3m)

    out = work / "out"
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(tgt),
           "--model", "boltz2", "--msa_dir", str(msa), "--out_dir", str(out),
           "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
           "--seed", "0", "--output_format", "cif"]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    if a.arm == "off":
        env.update({v: "0" for v in LEVERS})
    print(f"+ arm={a.arm} " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=7200)
    wall = time.perf_counter() - t0
    tail = "\n".join(r.stdout.splitlines()[-6:])
    print(f"rc={r.returncode} wall={wall:.3f}s\n{tail}", flush=True)
    res = {"arm": a.arm, "fixture": a.fixture, "rc": r.returncode, "wall_s": round(wall, 3),
           "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
           "levers_in_env": {v: env.get(v) for v in LEVERS}}
    if r.returncode != 0:
        print(r.stderr[-4000:])
        res["stderr_tail"] = r.stderr[-4000:]
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
        return 1

    cifs = sorted(out.rglob("*.cif"))
    assert len(cifs) == 1, f"expected one CIF, got {cifs}"
    k_cli, x_cli = read_atoms(cifs[0])
    res.update(n_atoms=len(x_cli),
               cif_sha256=hashlib.sha256(cifs[0].read_bytes()).hexdigest())
    print(f"CLI {a.arm} fold: {len(x_cli)} atoms  file sha256 {res['cif_sha256'][:16]}")
    if a.keep:
        a.keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], a.keep / cifs[0].name)
    rc = 0
    if a.ref:
        k_ref, x_ref = read_atoms(a.ref)
        same_keys = k_cli == k_ref
        same_xyz = bool(np.array_equal(x_cli, x_ref))
        res.update(ref=str(a.ref), ref_sha256=hashlib.sha256(a.ref.read_bytes()).hexdigest(),
                   atom_identity_identical=same_keys, coordinates_identical=same_xyz)
        print(f"ref {a.ref.name}: {len(x_ref)} atoms  file sha256 {res['ref_sha256'][:16]}")
        print(f"atom identity identical: {same_keys}   coordinates identical: {same_xyz}")
        if not (same_keys and same_xyz):
            d = float(np.abs(x_cli - x_ref).max()) if same_keys else float("nan")
            res["max_abs_delta_A"] = d
            print(f"MISMATCH, max |delta| = {d}")
            rc = 1
        else:
            print(f"PASS: the CLI {a.arm} arm reproduces {a.ref.name} exactly, end to end.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print("wrote", a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
