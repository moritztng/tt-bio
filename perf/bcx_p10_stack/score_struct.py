#!/usr/bin/env python3
"""Score the composed stack's structure against all-off, in Angstrom, on this fixture's own floor.

The campaign's 0.60 A kill bar is a PAIRED, same-stack, same-seed bar: two arms fold the same
complex from the same seeded logits, so everything but the arithmetic cancels. That is how these
arms are laid out -- `acc_off_a`, `acc_off_b` and `acc_on` are one fixture, one seed, one card.

    acc_off_a vs acc_off_b   the floor. The SAME code twice, so whatever it reads is what an
                             innocent re-run costs on this fixture and this metric.
    acc_off_a vs acc_on      the lever.

The 1.84 A constant is deliberately NOT used. It is Boltz-2's 512 aa diffusion fixture;
`perf/k10_anchor/FINDINGS.md` re-measured it as a 0.945-2.291 A distribution and
`perf/c12_orchestrator/landing/LANDING.md` retracts it campaign-wide, with the rule that every
accuracy claim carries a floor measured on its own fixture and metric. A BindCraft 2 seed does
not perturb a sampler, it draws a different design, so a cross-seed RMSD here would compare two
different molecules and mean nothing.

Read per round, because a gradient design loop compounds: round 0 is the two arms folding the
SAME sequence and is the arithmetic reading the bar is about; later rounds show how fast the
trajectories separate once the optimiser acts on a different gradient.

    python3 perf/bcx_p10_stack/score_struct.py perf/bcx_p10_stack/out --out acc_struct.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np


def read_cif(path):
    """atom_site rows out of a CIF BindCraft 2's own `write_structure` wrote."""
    cols, rows, in_loop, in_atom = [], [], False, False
    for line in pathlib.Path(path).read_text().splitlines():
        t = line.strip()
        if t == "loop_":
            cols, in_loop, in_atom = [], True, False
            continue
        if in_loop and t.startswith("_atom_site."):
            cols.append(t.split(".", 1)[1])
            in_atom = True
            continue
        if in_loop and t.startswith("_"):
            in_atom = False
            continue
        if in_atom:
            if not t or t.startswith("#") or t.startswith("loop_"):
                in_atom, in_loop = False, False
                continue
            f = t.split()
            if len(f) == len(cols):
                rows.append(dict(zip(cols, f)))
    if not rows:
        raise SystemExit(f"no _atom_site rows in {path}")
    return rows


def atoms(path):
    """{(chain, seq_id, atom_name): (xyz, plddt)} -- the key matches atom for atom.

    `write_structure` puts per-residue plDDT in B_iso_or_equiv, which is what lets a reading be
    restricted to the part of the prediction the model is actually committed to.
    """
    out = {}
    for r in read_cif(path):
        if r.get("group_PDB", "ATOM") not in ("ATOM", "HETATM"):
            continue
        key = (r.get("auth_asym_id") or r.get("label_asym_id"),
               r.get("label_seq_id") or r.get("auth_seq_id"),
               r.get("label_atom_id") or r.get("auth_atom_id"))
        try:
            out[key] = ((float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])),
                        float(r.get("B_iso_or_equiv", "nan")))
        except (KeyError, ValueError):
            continue
    return out


def kabsch_rmsd(p, q):
    """All-atom RMSD after optimal superposition. p, q are (N, 3) in matched order."""
    p = p - p.mean(0)
    q = q - q.mean(0)
    v, s, wt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(v @ wt))
    r = v @ np.diag([1.0, 1.0, d]) @ wt
    return float(np.sqrt((((p @ r) - q) ** 2).sum(1).mean()))


def pair(a, b, chain_filter=None, min_plddt=None, _cache={}):
    """RMSD over the atoms both structures actually have, so a missing atom cannot inflate it.

    `min_plddt` keeps only atoms BOTH arms call confident. At design round 1 the whole complex
    sits at mean plDDT ~52 with loops near 18, and an unconverged loop moves several Angstrom
    for any perturbation at all, so the whole-complex number is a chaos reading. The restricted
    one says whether the part the model is committed to agrees.
    """
    ka = _cache.setdefault(str(a), atoms(a))
    kb = _cache.setdefault(str(b), atoms(b))
    keys = sorted(set(ka) & set(kb))
    if chain_filter:
        keys = [k for k in keys if chain_filter(k[0])]
    if min_plddt is not None:
        keys = [k for k in keys if ka[k][1] >= min_plddt and kb[k][1] >= min_plddt]
    if len(keys) < 3:
        return None, len(keys)
    return (kabsch_rmsd(np.array([ka[k][0] for k in keys]),
                        np.array([kb[k][0] for k in keys])), len(keys))


def frames(root, tag):
    """{(stage, round, state): path} for one arm's per-round CIFs."""
    out = {}
    for p in pathlib.Path(root, tag).rglob("*.cif"):
        m = re.match(r"(.+)_(\d{4})_(.+)\.cif$", p.name)
        if m:
            out[(m.group(1), int(m.group(2)), m.group(3))] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base", default="acc_off_a")
    ap.add_argument("--floor", default="acc_off_b")
    ap.add_argument("--lever", default="acc_on")
    ap.add_argument("--binder-prefix", default="B",
                    help="chains starting with this are the designed binder")
    ap.add_argument("--min-plddt", type=float, default=70.0,
                    help="restrict one reading to atoms BOTH arms call this confident")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base, floor, lever = (frames(a.root, t) for t in (a.base, a.floor, a.lever))
    if not base:
        raise SystemExit(f"no frames under {a.root}/{a.base}")
    rows = []
    for k in sorted(set(base) & set(lever)):
        stage, rnd, state = k
        row = {"stage": stage, "round": rnd, "state": state}
        row["lever_all"], row["n_atoms"] = pair(base[k], lever[k])
        row["lever_binder"], row["n_binder"] = pair(
            base[k], lever[k], lambda c: c.startswith(a.binder_prefix))
        row["lever_conf"], row["n_conf"] = pair(base[k], lever[k], min_plddt=a.min_plddt)
        if k in floor:
            row["floor_all"], _ = pair(base[k], floor[k])
            row["floor_binder"], _ = pair(base[k], floor[k],
                                          lambda c: c.startswith(a.binder_prefix))
            row["floor_conf"], _ = pair(base[k], floor[k], min_plddt=a.min_plddt)
        row["plddt_base"] = round(float(np.nanmean(
            [v[1] for v in atoms(base[k]).values()])), 3)
        row["plddt_lever"] = round(float(np.nanmean(
            [v[1] for v in atoms(lever[k]).values()])), 3)
        rows.append(row)
    if not rows:
        raise SystemExit("no round matched between the arms")

    def g(r, k):
        v = r.get(k)
        return float("nan") if v is None else v

    print(f"{'rnd':>4}{'lever all':>11}{'floor all':>11}{'lever bind':>12}"
          f"{'lever conf':>12}{'floor conf':>12}{'n_conf':>8}"
          f"{'plddt base':>12}{'plddt lever':>13}")
    for r in rows:
        print(f"{r['round']:>4}{g(r,'lever_all'):>11.4f}{g(r,'floor_all'):>11.4f}"
              f"{g(r,'lever_binder'):>12.4f}{g(r,'lever_conf'):>12.4f}"
              f"{g(r,'floor_conf'):>12.4f}{r['n_conf']:>8}"
              f"{g(r,'plddt_base'):>12.3f}{g(r,'plddt_lever'):>13.3f}")
    first = rows[0]
    print()
    print(f"round {first['round']} is the only paired reading: from round 2 the two arms have "
          f"different sequences,\nso everything after it is the optimiser diverging, not the "
          f"arithmetic.\n"
          f"  lever  {g(first,'lever_all'):.4f} A all-atom, "
          f"{g(first,'lever_conf'):.4f} A over the {first['n_conf']} atoms both arms call "
          f"plDDT >= {a.min_plddt:.0f}\n"
          f"  floor  {g(first,'floor_all'):.4f} A all-atom, "
          f"{g(first,'floor_conf'):.4f} A confident\n"
          f"  plDDT  {g(first,'plddt_base'):.3f} -> {g(first,'plddt_lever'):.3f}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
