#!/usr/bin/env python3
"""p117 -- the ruler p106 never had.

p106 measured the block-sparse arm against the shipped arm and got sequence identity 0.9080 and
backbone RMSD 1.167 A, and p4 read that as "bifurcation, not tolerance" and decided against the
lever. That reading needs a scale it never had: RFD3 is a 200-step diffusion sampler, so a
sub-ULP perturbation is expected to produce a different design, and 0.9080 only means something
next to how far apart two SAMPLES of the same arm are.

This runs the same comparison across SEEDS instead of across arms, with the same instrument
(p106b), so the two are directly comparable. It also reports the fraction of backbone atoms that
are bit-identical, which is what makes an un-superposed RMSD legitimate here: the designed target
motif is frozen, identical in every output, so all four comparisons share a frame and none of them
needs a Kabsch fit.

Takes output directories written by p106 runs at two seeds. No device, no fold.
"""
import json
import pathlib
import sys

import torch

BACKBONE = ("N", "CA", "C", "O")
CIF = "R4_b100.cif"


def parse(path):
    """{(chain, resnum, atom): (x, y, z)} from the _atom_site loop. Same reader as p106b."""
    lines = path.read_text().splitlines()
    atoms, i = {}, 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        tags, j = [], i + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            tags.append(lines[j].strip())
            j += 1
        if not any(t.startswith("_atom_site.") for t in tags):
            i = j
            continue
        col = {t: k for k, t in enumerate(tags)}

        def pick(*names):
            for n in names:
                if n in col:
                    return col[n]
            raise KeyError(names)

        cx, cy, cz = (col["_atom_site.Cartn_" + a] for a in "xyz")
        cch = pick("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        crn = pick("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        can = col["_atom_site.label_atom_id"]
        while j < len(lines) and lines[j].strip() not in ("#", "loop_", ""):
            f = lines[j].split()
            if len(f) < len(tags):
                break
            atoms[(f[cch], f[crn], f[can])] = (float(f[cx]), float(f[cy]), float(f[cz]))
            j += 1
        i = j
    return atoms


def compare(a, b):
    A, B = parse(pathlib.Path(a) / CIF), parse(pathlib.Path(b) / CIF)
    keys = [k for k in A if k in B and k[2] in BACKBONE]
    d = torch.tensor([[A[k][i] - B[k][i] for i in range(3)] for k in keys]).norm(dim=1)
    frozen = int((d < 1e-6).sum())
    seq_a = {(k[0], k[1]) for k in A}
    return dict(
        n_backbone=len(keys),
        rmsd=round(float((d ** 2).mean().sqrt()), 4),
        max_shift=round(float(d.max()), 4),
        bit_identical=frozen,
        bit_identical_frac=round(frozen / len(keys), 4),
        n_moving=len(keys) - frozen,
        # the RMSD above averages over the frozen atoms too; this is the same number restricted
        # to the atoms that actually moved, which is what a reader means by "how different"
        rmsd_moving=round(float((d[d >= 1e-6] ** 2).mean().sqrt()), 4) if frozen < len(keys) else 0.0,
    )


if __name__ == "__main__":
    s42_off, s42_on, s43_off, s43_on = sys.argv[1:5]
    out = pathlib.Path(sys.argv[5] if len(sys.argv) > 5 else "perf/p117/seed_control.json")
    pairs = [
        ("arm delta, seed 42", s42_off, s42_on),
        ("arm delta, seed 43", s43_off, s43_on),
        ("seed 42 vs 43, shipped arm", s42_off, s43_off),
        ("seed 42 vs 43, sparse arm", s42_on, s43_on),
    ]
    rows = []
    for label, a, b in pairs:
        r = compare(a, b)
        r["pair"] = label
        r["kind"] = "arm" if label.startswith("arm") else "seed"
        rows.append(r)
        print("%-30s n=%d  frozen %.4f  RMSD %7.4f A  (moving-only %7.4f)  max %.4f"
              % (label, r["n_backbone"], r["bit_identical_frac"], r["rmsd"],
                 r["rmsd_moving"], r["max_shift"]))
    arm = [r["rmsd"] for r in rows if r["kind"] == "arm"]
    seed = [r["rmsd"] for r in rows if r["kind"] == "seed"]
    verdict = max(arm) < min(seed)
    print()
    print("arm-to-arm backbone RMSD  %.4f - %.4f A" % (min(arm), max(arm)))
    print("seed-to-seed backbone RMSD %.4f - %.4f A" % (min(seed), max(seed)))
    print("the arms differ by LESS than the seed does: %s" % verdict)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows=rows, arm_rmsd=arm, seed_rmsd=seed,
                                   arm_delta_below_seed_delta=verdict), indent=2) + "\n")
    print("wrote %s" % out)
