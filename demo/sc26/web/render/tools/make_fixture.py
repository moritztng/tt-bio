#!/usr/bin/env python3
"""Development fixture for the renderer: a real structure plus SYNTHETIC noise frames.

The frames are not a model's trajectory. They follow the EDM noise schedule Boltz-2 samples with
(sigma_max 160, sigma_min 4e-4, rho 7, sigma_data 16) applied to a known structure, with noise that
drifts slowly between steps and a random rotation per step like a real sampler's augmentation, so
the renderer's alignment, camera and transitions see realistic input before engine's real
trajectories land. The file says so in meta.source = "synthetic", and nothing at the booth may
show one: gallery trajectories replace them.

    make_fixture.py structure.cif out.json [--steps 200] [--seconds 6] [--seed 0]
"""
import argparse
import base64
import json
import shlex
from pathlib import Path

import numpy as np


def read_atoms(path):
    text = Path(path).read_text().splitlines()
    atoms = []
    if any(l.startswith("_atom_site.") for l in text):
        cols, rows, i = [], [], 0
        while i < len(text):
            if text[i].startswith("_atom_site."):
                while i < len(text) and text[i].startswith("_atom_site."):
                    cols.append(text[i].split()[0][len("_atom_site."):])
                    i += 1
                while i < len(text) and text[i].strip() and not text[i].startswith(("#", "_", "loop_")):
                    rows.append(shlex.split(text[i]))
                    i += 1
                break
            i += 1
        c = {k: j for j, k in enumerate(cols)}
        for r in rows:
            if r[c["group_PDB"]] != "ATOM":
                continue
            if "pdbx_PDB_model_num" in c and r[c["pdbx_PDB_model_num"]] != "1":
                continue
            alt = r[c["label_alt_id"]] if "label_alt_id" in c else "."
            if alt not in (".", "?", "A"):
                continue
            el = r[c["type_symbol"]]
            if el == "H":
                continue
            atoms.append(dict(name=r[c["label_atom_id"]].strip('"'), el=el, res=r[c["label_comp_id"]],
                              chain=r[c.get("auth_asym_id", c["label_asym_id"])],
                              seq=r[c.get("auth_seq_id", c["label_seq_id"])],
                              xyz=[float(r[c["Cartn_x"]]), float(r[c["Cartn_y"]]), float(r[c["Cartn_z"]])]))
    else:
        for l in text:
            if l.startswith("ENDMDL"):
                break
            if not l.startswith("ATOM"):
                continue
            if l[16] not in " A":
                continue
            el = (l[76:78].strip() or l[12:16].strip()[0])
            if el == "H":
                continue
            atoms.append(dict(name=l[12:16].strip(), el=el, res=l[17:20], chain=l[21], seq=l[22:27].strip(),
                              xyz=[float(l[30:38]), float(l[38:46]), float(l[46:54])]))
    return atoms


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    atoms = read_atoms(a.structure)
    res_key, res_name, chain, ridx = {}, [], [], []
    for at in atoms:
        k = (at["chain"], at["seq"])
        if k not in res_key:
            res_key[k] = len(res_name)
            res_name.append(at["res"])
            chain.append(at["chain"])
        ridx.append(res_key[k])
    x0 = np.array([at["xyz"] for at in atoms], dtype=np.float64)
    x0 -= x0.mean(0)

    rng = np.random.default_rng(a.seed)
    smax, smin, rho, n = 160.0, 4e-4, 7.0, a.steps
    i = np.arange(n)
    sig = (smax ** (1 / rho) + i / (n - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    eps = rng.normal(size=x0.shape)
    frames = []
    for k in range(n + 1):
        if k < n:
            eps = np.sqrt(0.9) * eps + np.sqrt(0.1) * rng.normal(size=x0.shape)
            x = x0 + sig[k] * eps
            x = (x - x.mean(0)) @ random_rotation(rng).T
        else:
            x = x0  # the final frame is the structure itself
        frames.append(dict(t=round(a.seconds * k / n, 4), progress=round(k / n, 4),
                           coords=base64.b64encode(x.astype("<f4").tobytes()).decode()))
    doc = dict(
        meta=dict(source="synthetic", structure=Path(a.structure).name, natom=len(atoms), nres=len(res_name),
                  note="SYNTHETIC noise on a known structure, for renderer development only. Not a model trajectory."),
        topology=dict(atom_name=[at["name"] for at in atoms], element=[at["el"] for at in atoms],
                      residue_index=ridx, res_name=res_name, chain=chain),
        frames=frames)
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"{a.out}: {len(atoms)} atoms, {len(res_name)} residues, {len(frames)} frames")


if __name__ == "__main__":
    main()
