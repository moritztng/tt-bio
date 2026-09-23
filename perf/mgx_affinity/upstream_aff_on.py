#!/usr/bin/env python3
"""Upstream Boltz-2's affinity stage on a structure TT predicted, on CPU, no device opened.

    <upstream venv>/bin/python upstream_aff_on.py <input.yaml> <tt.cif> <out_dir> [--seed N] [--threads T]

Upstream's affinity pass reads the top-ranked structure from `predictions/<id>/pre_affinity_<id>.npz`
and crops its pocket from those coordinates. This writes that file from TT's mmCIF (upstream's own
processed structure with TT's coordinates, matched by chain, residue and atom name) and then runs
`boltz predict` at predict's affinity settings, so the structure stage is skipped and the affinity
heads see exactly the pose TT's own affinity leg saw. A 1536 aa target is out of reach for an
upstream CPU fold; this is not, because the affinity pass is a fixed 256-token crop.
"""
import argparse
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import gemmi
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("cif")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()

from boltz.data.types import Coords, Interface, Manifest, StructureV2  # noqa: E402
from boltz.main import process_inputs  # noqa: E402

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
stem = Path(a.input).stem
yaml = out / f"{stem}.yaml"
# `msa: empty` is upstream's spelling of predict's --single_sequence.
yaml.write_text("".join(ln + ("      msa: empty\n" if ln.startswith("      sequence: ") else "")
                        for ln in Path(a.input).read_text().splitlines(keepends=True)))
res = out / f"boltz_results_{stem}"
cache = Path.home() / ".boltz"
process_inputs(data=[yaml], out_dir=res, ccd_path=cache / "ccd.pkl", mol_dir=cache / "mols",
               msa_server_url="", msa_pairing_strategy="greedy", boltz2=True)
(record,) = Manifest.load(res / "processed" / "manifest.json").records

s = StructureV2.load(res / "processed" / "structures" / f"{record.id}.npz").remove_invalid_chains()
tt = {}
for ch in gemmi.read_structure(a.cif)[0]:
    for i, r in enumerate(ch):
        for at in r:
            tt[(ch.name, i, at.name)] = at.pos.tolist()
atoms = s.atoms.copy()
coords = np.empty((len(atoms), 3), dtype=np.float32)
for c in s.chains:
    for ri, r in enumerate(s.residues[c["res_idx"]:c["res_idx"] + c["res_num"]]):
        for k in range(r["atom_idx"], r["atom_idx"] + r["atom_num"]):
            coords[k] = tt[(str(c["name"]), ri, str(atoms[k]["name"]))]
if len(tt) != len(atoms):
    sys.exit(f"atom count differs: TT cif {len(tt)} vs upstream structure {len(atoms)}")
atoms["coords"] = coords
atoms["is_present"] = True
residues = s.residues.copy()
residues["is_present"] = True
new = replace(s, atoms=atoms, residues=residues, interfaces=np.array([], dtype=Interface),
              coords=np.array([(x,) for x in coords], dtype=Coords))
pred = res / "predictions" / record.id
pred.mkdir(parents=True, exist_ok=True)
np.savez_compressed(pred / f"pre_affinity_{record.id}.npz", **asdict(new))
print(f"pre_affinity written: {len(atoms)} atoms from {a.cif}", flush=True)

# predictions/<id>/ exists, so the structure stage skips this record and only affinity runs.
env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
sys.exit(subprocess.call(
    [str(Path(sys.executable).parent / "boltz"), "predict", str(yaml), "--out_dir", str(out),
     "--seed", str(a.seed), "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
     "--accelerator", "cpu", "--no_kernels", "--preprocessing-threads", "1", "--cache", str(cache)],
    env=env))
