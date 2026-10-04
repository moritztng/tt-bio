# ESMFold2 on 1x NVIDIA H200 -- throughput for the sc26 booth comparison (2026-10-04)

Target: HSA[:300]. That is exactly the first 300 characters of the given 349-char string, ending `...AEVENDEMPA`, the same slicing demo/sc26/engine/bench_live.py uses. Single chain, seed 0, 1 diffusion sample.
Primary config: num_loops=3, num_sampling_steps=20. Secondary: 10/100, the gpu5_bench esmfold2 default.
Method: esm_conc.py, which mirrors gpu_concurrency.py. N independent processes, each with its own CUDA context. Each runs 1 untimed warm fold, then all meet at a file barrier, then run the timed folds. folds/s = total timed folds / (last timed end - first timed start). Fold path mirrors gpu5_bench.run_esmfold2: from_pretrained().cuda().eval(), manual_seed(0), no_grad, infer_protein_as_pdb. GPU stats come from nvidia-smi sampled every 0.5 s inside the timed window.

| N | mode | config | folds/s | median latency s | util % (mean) | power W (mean/max) | GPU mem MiB |
|---|------|--------|--------:|-----------------:|--------------:|-------------------:|------------:|
| 1 | plain | 3/20 (5 timed) | 0.2943 | 3.398 | 97.2 | 611 / 648 | 28,783 |
| 2 | MPS | 3/20 (4/proc) | 0.3301 | 6.041 | 99.7 | 684 / 701 | 57,618 |
| 4 | MPS | 3/20 (4/proc) | **0.3334** | 11.981 | 100.0 | 694 / 701 | 115,168 |
| 8 | MPS | 3/20 | OOM -- all 8 workers CUDA OOM at load (fp32, ~28 GB/process; 8 x 28 > 140 GB) | - | - | - | - |
| 1 | plain | 10/100 (5 timed) | 0.1076 | 9.295 | 96.8 | 612 / 665 | 28,783 |
| 4 | MPS | 10/100 (4/proc) | 0.1232 | 32.459 | 100.0 | 692 / 701 | 115,169 |

Best sustained throughput for the primary config: **0.333 folds/s (about 1,200 folds/h)** at MPS N=4. That is 1.13x the N=1 rate. N=1 already reaches 97% SM utilisation, so the GPU is close to saturated from the start. N=4 is the largest N that fits in memory. N=12 was not run: it cannot fit.

## Provenance / deviations (read these)
- **Not the published cell's software stack.** The published H200 cell used transformers 4.57.6 + esm @26b0bc2b, the Biohub transformers fork with ESMFold2Model, and HF revision 8fc3ff47. That stack can no longer be rebuilt. esm@26b0bc2b depends on github.com/Biohub/transformers, which now returns 404. PyPI transformers 4.57.6 has no esmfold2 module.
  - Instead I used upstream ESMFold2 as it ships today: transformers 5.18.0 `EsmFold2Model`, loading biohub/ESMFold2 at hub main, snapshot 69869f737beffec5294845ede23db5fc0b4f509e. That snapshot is the re-uploaded HF-native schema that bundles the ESMC-6B weights.
  - The old pins (8fc3ff47 / 45b0fa5d) are not loadable with this code.
- Precision: fp32. All parameters are torch.float32, because config `dtype: float32` is the from_pretrained default. That matches the published path, which also used default from_pretrained (fp32). No autocast or bf16 was applied.
- Kernel backend: transformers 5 has no `set_kernel_backend`/cuequivariance switch. I tested `use_kernels=True` (fused Triton hub kernel for triangle-multiplication) in the smoke runs: 3.409 s against 3.399 s for the reference. No measurable difference, so the reference path was used. I did not confirm that the hub kernel actually engaged.
- cuDNN SDPA was left enabled. That is the gpu5_bench behaviour on sm_90.

## Environment
GPU: NVIDIA H200 (143,771 MiB, 700 W limit). Driver 595.71.05. torch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24). transformers 5.18.0. cuequivariance 0.12.0 (installed, unused by this code path). Python 3.12.
Host: vast.ai machine 51172 (Czechia), Xeon Platinum 8568Y+, cgroup quota 23.04 vCPU (offer listed 24). Image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime. Full package list: pip_freeze_hf5.txt.
Instance 54156933. Interruptible bid at $1.60/hr; billed dph_total was $1.670/hr including disk. Created 14:00:29Z, destroyed 14:13:15Z UTC: about 13 min, about $0.36 of compute.
Account credit fell from $16.905 to $15.276 (-$1.63) over the job window. The gap to $0.36 is most likely storage on the pre-existing stopped instance 52151627 ("mgx-reference") plus billing granularity. I did not verify this. That instance was not created by this job and was left untouched.

Files: p_*.json (primary), s_*.json (secondary), logs_*/ (worker logs + w0.pdb), smoke_*.json, versions.txt, nvidia-smi*.txt, session.log, setup/session/esm_conc scripts, hsa300.seq.
