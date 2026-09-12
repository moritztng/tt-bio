# `TT_BIO_FUSE_BIAS_STACKS` at 512 aa

The flag is on by default and is not bit-exact. Its accuracy evidence was one 298-residue control.
This is the 512 aa score it never had: four seeds per arm on whglx card 3 (Wormhole B0, 8x9 grid),
200 sampling steps, 3 recycles, full 35-row MSA, one sample, templates off.

**Verdict: keep it on.** At 512 aa the fused arm moves each pseudo-domain 0.27-0.49 A all-atom
against a 0.60 A bar, and the same arm re-folded at a different seed moves 0.97-1.91 A. The lever's
worst move is half the sampler's smallest one. Structure quality against the experimental CDK2
structure (1HCL) is unchanged: mean CA-lDDT 0.9377 -> 0.9370 on copy 1 and 0.9154 -> 0.9143 on
copy 2, inside a per-seed spread of 0.028.

## Why the fixture needs three readings

`cdk2x2_512` is CDK2 fused to its own first 214 residues with no interface between the copies, so
the hinge dominates any whole-molecule number. A seed change alone moves it 6.8-17.4 A whole
molecule and rotates the hinge by 53-158 degrees while leaving both domains intact. Whole-molecule
RMSD on this fixture is a hinge angle in disguise.

| reading | lever (on vs off, same seed) | seed floor (same arm, 12 pairs) |
|---|---|---|
| domain 1, all-atom | 0.316 - 0.470 A | 1.090 - 1.906 A |
| domain 2, all-atom | 0.271 - 0.491 A | 0.967 - 1.634 A |
| CA-lDDT between the two structures | 0.9874 - 0.9986 | 0.887 - 0.939 |
| hinge | 0.8 - 10.8 deg | 52.6 - 157.5 deg |
| whole molecule, all-atom | 0.341 - 1.532 A | 6.80 - 17.36 A |

The 298 aa control reproduces the number main ships on: 0.2198 A all-atom at seed 0, 0.175-0.220 A
across four seeds, against a 0.83-1.25 A seed floor.

The A/A floor is exactly zero: the first fold repeated as the last one comes back byte-identical at
both sizes, and the seed-0 unfused CIF carries the digest `2e3b25a5520c8ff8` that
`perf/b2z_levers/wh_neutral_whglx_c3.json` recorded for the same arm on this card at an earlier
commit.

## What the flag actually perturbs

`_fuse_bias_stack` is host torch, not a kernel: one affine-free `layer_norm` plus one wide `linear`
in place of a per-layer LayerNorm+Linear loop. On Boltz-2's own weights that is 8.58e-06 max and
2.55e-07 mean absolute on the 24-layer token stack, 5.59e-06 relative to the output RMS - a few
fp32 ULPs entering the diffusion conditioning. The arithmetic is the same on any host CPU, so an
architecture can only change what the device does with the perturbed conditioning downstream.

## Files

* `fold_seeds.py` - both arms at four seeds plus an A/A repeat, one process, one device open.
* `score.py` - per pseudo-domain RMSD and hinge, CA-lDDT arm-against-arm and against 1HCL.
  Superposition, parser and lDDT are `perf/other512`, `perf/b2x-flag-levers` and `perf/fused_sdpa`'s.
* `host_probe.py` - the perturbation above, CPU only, real weights.
* `seeds_whglx_c3.json`, `score_whglx_c3.json`, `host_probe.json`, `cif/` - the run and its reading.

    python3 perf/b2z2_fusebias/fold_seeds.py --out seeds.json --cifdir cif --seeds 0,1,2,3
    python3 perf/b2z2_fusebias/score.py cif --runs seeds.json --split 298 --out score.json
