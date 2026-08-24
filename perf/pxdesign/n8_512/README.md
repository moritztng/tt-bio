# PXDesign's batching term at 512 residues

The GPU reference has PXDesign at `N_sample` 8 for a 116-residue target (5.254x per design) and for
768 (1.434x), and not at 512, which is the size the perf page publishes. The term does not
interpolate between those two, so it was measured.

One rented H200 on the reference's own pin (`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`, torch
2.3.1 / CUDA 12.1), `gpu_pxdesign_setup.sh` unmodified. Two arms in one session on one exclusive
card, three reps each with rep 0 dropped cold, `preview` preset, 400 steps, bf16, seed 42, the same
`laczc_512` target; only `--N_sample` differs. `laczc_512_nomsa.yaml` is `targets/laczc_512.yaml`
with the `msa:` key removed, which makes it the yaml behind the published `laczc512_prev_n1` cell.

| arm | generator wall, warm median | per design | gen_device | utilisation | power |
|---|---|---|---|---|---|
| N_sample 1 | 32.4746 s (0.90 % spread) | 32.4746 s | 27.7841 s | 70.9 % | 329.0 W |
| N_sample 8 | 123.2831 s (0.36 % spread) | 15.4104 s | 118.5885 s | 98.9 % | 423.8 W |

2.1073x on this box. The published cell carries the more conservative **1.9757x, 15.5962 s per
design**: this box's host featurisation ran 4.60 s against the published cell's 2.1528 s, which
inflates a wall amortisation, so only the device ratio (4.2682x for eight samples) is transferred
onto the published cell's own host timings. The device half reproduces the published cell to 2.8 %,
27.7841 s against 28.5825 s, which is what makes the transfer legitimate.

`indep_check.py` answers the question that decides whether any of this is a throughput gain: are the
eight independent designs, or one design eight times. Eight 80-residue binder backbones, 28 pairs,
zero identical, pairwise CA RMSD 17.68 / 54.64 / 73.68 A min/median/max, identical across all three
reps. Quality does not degrade against the N=1 arm on any metric AF2-IG reports: median af2_plddt
0.935 against 0.93, monomer pLDDT 0.955 against 0.955, monomer pTM 0.813 against 0.789, ipAE 27.23
against 27.31, binder self-consistency RMSD 0.50 A against 0.77. AF2-IG binding success is false on
both arms (0 of 24 and 0 of 3) because this fixture is a cost fixture with an arbitrary epitope, so
that flag cannot discriminate here and is not read as if it could.

Nothing equivalent has been measured on Tenstorrent. Until it is, the row stays off the per-server
and per-dollar charts: giving one side a lever and not the other is why RFdiffusion3 is off them.

`pxd_final.jsonl` is the six per-rep records, `summaries.txt` the per-design AF2-IG metrics of the
four warm reps, `chain_final_run.txt` the run as it happened (renamed off `.log`, which the repo gitignores).
