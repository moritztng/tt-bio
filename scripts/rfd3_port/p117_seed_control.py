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
FROZEN = 1e-6      # a backbone atom that moved less than this is motif, not design


def parse(path):
    """{(chain, resnum, atom): (x, y, z)} from the _atom_site loop. Same reader as p106b."""
    lines = path.read_text().splitlines()
    atoms, names, i = {}, {}, 0
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
        cco = col["_atom_site.label_comp_id"]
        while j < len(lines) and lines[j].strip() not in ("#", "loop_", ""):
            f = lines[j].split()
            if len(f) < len(tags):
                break
            atoms[(f[cch], f[crn], f[can])] = (float(f[cx]), float(f[cy]), float(f[cz]))
            names[(f[cch], f[crn])] = f[cco]
            j += 1
        i = j
    return atoms, names


def moving_residues(a, b):
    """Residues whose backbone actually moved between two outputs: the designed positions."""
    (A, _), (B, _) = parse(pathlib.Path(a) / CIF), parse(pathlib.Path(b) / CIF)
    keys = [k for k in A if k in B and k[2] in BACKBONE]
    d = torch.tensor([[A[k][i] - B[k][i] for i in range(3)] for k in keys]).norm(dim=1)
    return {(k[0], k[1]) for k, dd in zip(keys, d.tolist()) if dd >= FROZEN}


def compare(a, b, designed=None):
    (A, na), (B, nb) = parse(pathlib.Path(a) / CIF), parse(pathlib.Path(b) / CIF)
    keys = [k for k in A if k in B and k[2] in BACKBONE]
    d = torch.tensor([[A[k][i] - B[k][i] for i in range(3)] for k in keys]).norm(dim=1)
    frozen = int((d < FROZEN).sum())
    # Sequence identity over ALL residues is 85 % frozen motif, identical by construction, which
    # compresses every comparison into the same narrow band around 0.9. The number that carries
    # information is identity over the positions the model actually designs.
    shared = [k for k in na if k in nb]
    all_diff = sum(1 for k in shared if na[k] != nb[k])
    des = [k for k in shared if designed is None or k in designed]
    des_diff = sum(1 for k in des if na[k] != nb[k])
    return dict(
        n_backbone=len(keys),
        rmsd=round(float((d ** 2).mean().sqrt()), 4),
        max_shift=round(float(d.max()), 4),
        bit_identical=frozen,
        bit_identical_frac=round(frozen / len(keys), 4),
        n_moving=len(keys) - frozen,
        # the RMSD above averages over the frozen atoms too; this is the same number restricted
        # to the atoms that actually moved, which is what a reader means by "how different"
        rmsd_moving=round(float((d[d >= FROZEN] ** 2).mean().sqrt()), 4) if frozen < len(keys) else 0.0,
        n_residues=len(shared),
        seq_identity=round(1 - all_diff / len(shared), 4),
        n_designed=len(des),
        designed_differing=des_diff,
        designed_identity=round(1 - des_diff / len(des), 4) if des else None,
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
    # Score every pair on the SAME denominator: the residues that move in any comparison. That
    # set is the design task; the rest is the frozen motif.
    designed = set()
    for _, a, b in pairs:
        designed |= moving_residues(a, b)
    print("designed positions (move in any comparison): %d\n" % len(designed))
    rows = []
    for label, a, b in pairs:
        r = compare(a, b, designed)
        r["pair"] = label
        r["kind"] = "arm" if label.startswith("arm") else "seed"
        rows.append(r)
        print("%-27s RMSD %7.4f A (moving %7.4f)  seq id %.4f  DESIGNED id %.4f (%d/%d differ)"
              % (label, r["rmsd"], r["rmsd_moving"], r["seq_identity"],
                 r["designed_identity"], r["designed_differing"], r["n_designed"]))
    arm = [r["rmsd"] for r in rows if r["kind"] == "arm"]
    seed = [r["rmsd"] for r in rows if r["kind"] == "seed"]
    arm_id = [r["designed_identity"] for r in rows if r["kind"] == "arm"]
    seed_id = [r["designed_identity"] for r in rows if r["kind"] == "seed"]
    verdict = max(arm) < min(seed) and min(arm_id) > max(seed_id)
    print()
    print("arm-to-arm backbone RMSD  %.4f - %.4f A" % (min(arm), max(arm)))
    print("seed-to-seed backbone RMSD %.4f - %.4f A" % (min(seed), max(seed)))
    print("arm-to-arm designed identity   %.4f - %.4f" % (min(arm_id), max(arm_id)))
    print("seed-to-seed designed identity  %.4f - %.4f" % (min(seed_id), max(seed_id)))
    print("the arms differ by LESS than the seed does, on BOTH axes: %s" % verdict)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows=rows, n_designed=len(designed),
                                   arm_rmsd=arm, seed_rmsd=seed,
                                   arm_designed_identity=arm_id, seed_designed_identity=seed_id,
                                   arm_delta_below_seed_delta=verdict), indent=2) + "\n")
    print("wrote %s" % out)
