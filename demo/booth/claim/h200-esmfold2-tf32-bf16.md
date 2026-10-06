# ESMFold2 on 1x H200: TF32 / bf16 rungs (booth follow-up, 2026-10-04)

The config and method are the same as gpu/SUMMARY.md: HSA[:300] (hsa300.seq), num_loops=3, num_sampling_steps=20, 1 diffusion sample, seed 0. Software is transformers 5.18.0 EsmFold2Model with biohub/ESMFold2 snapshot 69869f73 (pinned through HF_REV). The env was rebuilt from gpu/pip_freeze_hf5.txt: torch 2.14.1+cu130, cuDNN 9.24, kernels 0.17.2, Python 3.12. gcc had to be apt-installed because the ESMC rotary embedding JIT-builds a Triton kernel.
Script: esm_prec.py (= gpu/esm_conc.py plus a PREC switch). Plain runs: 1 warm fold + 5 timed folds, median latency. MPS runs: N processes, each 1 warm + 4 timed; folds/s = total timed folds / span.
Accuracy: acc.py computes CA RMSD after Kabsch superposition against gpu/logs_p_plain_n1/w0.pdb (the fp32 seed-0 fold from the earlier run) and mean CA pLDDT. Every warm-fold and last-timed-fold PDB of every worker was compared (accuracy.json, accuracy_mps.json).
The fp32 control in this session reproduced the reference at 0.000 A (bit-identical) and ran at 0.2946 folds/s, against 0.2943 in the earlier run.

| rung | N | folds/s | median latency s | CA RMSD vs fp32 (A) | mean pLDDT | peak GPU mem |
|---|---|---:|---:|---:|---:|---:|
| fp32 control | 1 plain (3 timed) | 0.2946 | 3.393 | 0.000 | 92.90 | 28.8 GB |
| A TF32 matmul+cudnn | 1 plain | 0.5330 | 1.876 | 0.011-0.013 | 92.89-92.90 | 28.8 GB |
| A TF32 | 2 MPS | 0.5402 | 3.697 | 0.010-0.014 | 92.88-92.90 | 57.6 GB |
| A TF32 | 4 MPS | **0.5444** | 7.329 | 0.009-0.013 | 92.88-92.90 | 115 GB |
| B bf16 autocast (fp32 weights) | 1 | FAILS TO RUN | - | - | - | - |
| B' native bf16: from_pretrained(dtype=bf16) | 1 plain | 0.5790 | 1.727 | 0.472-0.537 | 92.40-92.51 | 15.0 GB |
| B' native bf16 | 2 MPS | 0.6256 | 3.180 | 0.462-0.533 | 92.46-92.51 | 30.1 GB |
| B' native bf16 | 4 MPS | **0.6394** | 6.228 | 0.491-0.535 | 92.46-92.50 | 60.2 GB |
| B' native bf16 | 8 MPS | 0.6384 | 12.482 | 0.432-0.539 | 92.42-92.55 | 120 GB |
| C model.to(bf16) | 1 | FAILS TO RUN | - | - | - | - |

All rungs that ran pass the accuracy bar (RMSD < 1.0 A, pLDDT drop < 2). TF32 is practically exact. Native bf16 costs about 0.5 A and about 0.45 pLDDT. For scale: fp32 at 10/100 against fp32 at 3/20 differs by 0.85 A.

**Fastest passing rung: native bf16 (`from_pretrained(dtype=torch.bfloat16)`), best sustained 0.639 folds/s (about 2,300 folds/h) at MPS N=4. N=8 is flat at 0.638.** That is 2.17x fp32 plain N=1 and 1.92x the previous fp32 best (MPS N=4, 0.3334). The best TF32 figure is 0.544 folds/s (1.63x the fp32 best).

## What each rung actually did
- A (TF32): the profiler shows `sm90_xmma_gemm_f32f32_tf32f32_f32` cuBLAS kernels. Params are all fp32 and matmul_allow_tf32=True.
- B (plain bf16 autocast around infer_protein_as_pdb, fp32 weights): crashes upstream in generation_esmfold2._weighted_rigid_align with `NotImplementedError: "svd_cuda_gesvd" not implemented for 'BFloat16'`. Autocast reaches the diffusion rigid-align SVD. Making it run would need a code patch (autocast disabled around the SVD), and I did not patch. Log: results/logs_ac_plain_n1/w0.log.
- B' (the upstream native bf16 option): `from_pretrained(dtype=torch.bfloat16)` gives 1435 bf16 + 1220 fp32 params. The fp32 ones are `_keep_in_fp32_modules_strict`: norms, fourier, distogram head. ESMC runs under its own internal bf16 autocast (modeling_esmfold2.py:2025). The profiler shows `cutlass_80_tensorop_bf16_s16816gemm` kernels. Peak reserved is 13.9 GiB per process, so N=8 fits.
- C (whole-model `.to(torch.bfloat16)`): crashes in layer_norm with `expected scalar type Float but found BFloat16`. Upstream does not support it. Log: results/logs_bft_plain_n1/w0.log.
- use_kernels=True (fused Triton TriMul hub kernel) on top of bf16: NOT tested because the time cap ran out.

## Environment / cost / provenance
GPU: NVIDIA H200 143,771 MiB, driver 595.71.05, vast.ai machine 51172 (Czechia), offer 29522438. Instance 54158546, interruptible.
- Created 14:15:52Z at a $1.60/hr bid (dph_total $1.688).
- **Preempted (stopped) around 14:20-14:29Z** in the middle of the TF32 rung. The bid was raised to $2.40/hr (dph_total $2.488) and the instance restarted at 14:30:21Z. Rungs A and B' ran after the restart. The partial TF32 run before the preemption produced no result.
- Destroyed 14:37:14Z. Verified absent at 14:37:20Z. Instance 52151627 (mgx-reference) was untouched.
- Compute cost: at most about $0.89 (21.1 min wall x $2.49/hr upper bound). Actual running time was about 11 min.
- Account credit fell from $15.215 to $13.451 (-$1.76) over the window. As in the gpu/ run, that drop also includes charges I did not attribute (most likely mgx-reference storage), and I did not verify this.

Times are UTC. Files: results/*.json (per-run, incl. profiler top-40 kernels for plain runs), results/logs_*/ (worker logs + warm/last PDBs), results/session.log, results/setup.log, results/versions.txt, results/nvidia-smi*.txt, accuracy*.json, esm_prec.py, acc.py, setup.sh, session.sh, INSTANCE, instance_final.json.
