#!/usr/bin/env python3
"""What the SHIPPED DEFAULT writes at 512 aa, through the real CLI and its worker spawn.

The A/B harness sets the arms in-process (an env pop and a module assignment), which proves the
code paths and not the defaults a user gets. The published cell's parity sentence is a claim about
the CLI, so it has to be taken through the CLI: no lever variable in the environment, one fresh
process, `python3 -m tt_bio.main predict` with the cell's own protocol.

Records the CIF sha256-16, every confidence key results.json carries, and the coordinates, and
compares them against a CIF from the interleaved run (--against) when one is given.

    cli_default.py [--against <cif>] --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "other512"))
from cif_rmsd import kabsch_rmsd, read_atoms                                  # noqa: E402

FIX = REPO / "perf" / "size512" / "fixtures"
LEVERS = ("TT_BIO_FUSE_BIAS_STACKS", "TT_BIO_SDPA_ADD_GRANULARITY", "TT_BIO_GATE_GRANULARITY",
          "TT_BIO_HOST_LEVERS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", type=Path)
    ap.add_argument("--arm", choices=("ship", "base"), default="ship",
                    help="ship reads the defaults; base forces all three levers off in the child, "
                         "which is the pre-merge path, and says what the CLI wrote before the merge.")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    for v in LEVERS:
        assert v not in os.environ, f"{v} is set; this pass must read the DEFAULT"
    sys.path.insert(0, str(REPO))
    import tt_bio
    assert Path(tt_bio.__file__).resolve().is_relative_to(REPO), tt_bio.__file__
    from tt_bio import boltz2 as B2
    from tt_bio import reblock_permute as RB
    defaults = {"fuse_bias_stacks": bool(B2._fuse_bias_stacks()),
                "gate_granularity": RB.GATE_GRANULARITY}
    assert defaults["fuse_bias_stacks"] is True and defaults["gate_granularity"] == 2, defaults
    arm_env = {} if a.arm == "ship" else {"TT_BIO_FUSE_BIAS_STACKS": "0",
                                          "TT_BIO_SDPA_ADD_GRANULARITY": "1",
                                          "TT_BIO_GATE_GRANULARITY": "1"}
    from tt_bio.main import _read_bio_chains

    work = Path(tempfile.mkdtemp(prefix="b2z-recell-cli-"))
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
    env = dict(os.environ, PYTHONPATH=str(REPO), **arm_env)
    print("+", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    print(f"rc={r.returncode} wall={wall:.3f}s", flush=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        return 1

    cifs = sorted(out.rglob("*.cif"))
    assert len(cifs) == 1, f"expected one CIF, got {cifs}"
    body = cifs[0].read_bytes()
    sha = hashlib.sha256(body).hexdigest()[:16]
    conf = {}
    for j in sorted(out.rglob("results.json")):
        conf[str(j.relative_to(out))] = json.loads(j.read_text())
    rec = {"arm": a.arm, "arm_env": arm_env, "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
           "defaults": defaults, "cli_wall_s": round(wall, 3), "sha256": sha,
           "n_atoms": len(read_atoms(cifs[0])[1]), "results_json": conf,
           "cmd": cmd[1:], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.against:
        k_cli, x_cli = read_atoms(cifs[0])
        k_ab, x_ab = read_atoms(a.against)
        same = k_cli == k_ab
        rec["against"] = {
            "cif": str(a.against),
            "sha256": hashlib.sha256(a.against.read_bytes()).hexdigest()[:16],
            "atom_identity_identical": same,
            "coordinates_identical": bool(same and np.array_equal(x_cli, x_ab)),
            "max_abs_delta_A": (float(np.abs(x_cli - x_ab).max()) if same else None),
            "kabsch_all_atom_A": (round(kabsch_rmsd(x_cli, x_ab), 6) if same else None)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "results_json"}, indent=1))
    print("confidence:", json.dumps(conf, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
