# Provenance of every CIF in this directory

All eight structures below are the same target family at 3 recycles / 200 sampling steps / 1
diffusion sample / seed 0, templates off, MSA = `perf/size512/fixtures/cdk2x2_<size>.a3m` (35 rows).
Copied here, not regenerated, so `score.py` runs from this branch alone.

| file | where it came from |
|---|---|
| `512_on_1.cif`, `512_on_2.cif` | `wk/openfold3-to-4x:perf/of3_4x/ab/rmsd/512_on_{1,2}/cdk2x2_512.cif` — shipped path, detached worktree at `origin/main 8fc678b5`, card 3 |
| `512_P_1.cif`, `512_P_2.cif` | same tree, the fused-SDPA arm (pre-scaled pair bias, persistent-mask fused SDPA at HiFi4 + fp32_dest_acc) |
| `298_on_1.cif`, `298_P_1.cif` | `wk/openfold3-to-4x:perf/of3_4x/ab298/rmsd/298_{on,P}_1/cdk2x2_298.cif` |
| `512_refH200.cif` | `state/gpu5_512aa/h200/struct_openfold3_h200_seed0.cif` — official openfold3 0.4.4, torch 2.13.0+cu130, cuequivariance-ops-torch 0.11.1, H200, ckpt `of3-p2-155k.pt`, 2026-08-12 |
| `512_refB200.cif` | `state/gpu5_512aa/b200/struct_openfold3_b200_seed0.cif` — same stack on a B200, same seed |
| `1hcl.cif` | RCSB, `https://files.rcsb.org/download/1HCL.cif`. Human CDK2 apo; 294 resolved CA of 298, identical sequence to the fixture at 1:1 numbering |

The two writers (tt-bio's and upstream OF3's) emit the same 4116 atoms in the same order under the
same `(label_asym_id, label_seq_id, label_atom_id, label_comp_id)` keys, so no atom matching or
re-ordering happens anywhere in the scoring path.
