# rfd3-gpu-h200-b200 — live rental log

Append-only. **A relaunch reads this FIRST** to find a box that is still burning money, and to avoid
re-paying for an install that already happened.

## Live boxes

| box | vast instance | ssh | offer | $/hr | rented (UTC) | torn down |
|---|---|---|---|---|---|---|
| H200 | **47635549** | `-p 35548 root@ssh7.vast.ai` | 39605030 | 3.9351 | 2026-08-13 14:20 | **NO — live** |

Credit before renting: **$34.3722** (`vastai show user --raw`, agrees with the fixture doc's
post-teardown reading).

Teardown, when the H200 leg is done:

```bash
V=~/.vast-venv/bin/vastai
$V destroy instance 47635549
$V show instances-v1 --raw | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['instances'] if isinstance(d,dict) else d))"
```

`show instances-v1 --raw` returns a **dict** with an `instances` key, not a list. Indexing it as a
list throws `TypeError: string indices must be integers` on every poll and burns rental time.

## H200 — state of the box

Everything below is measured on instance 47635549.

| fact | value |
|---|---|
| GPU | NVIDIA H200, 143771 MiB, driver 580.159.03, sm_90 |
| `power.limit` | **700.00 W**, measured, not assumed |
| idle power | **78.7 W**, median of n = 100 at 200 ms, spread 78.59-78.98, util 0 %, 0 MiB used |
| host | 224 vCPU, 120 GB disk, python 3.11.13, base torch 2.8.0+cu129 |
| arm A venv | `/work/v_head` — py3.12, rc-foundry 0.2.0 + `attention.py` from foundry `4010e3e2e`, torch **2.13.0+cu130** |
| arm B venv | `/work/v_pip` — py3.12, `rc-foundry[rfd3]==0.2.0`, torch 2.13.0+cu130 |
| checkpoint | `/root/.foundry/checkpoints/rfd3_latest.ckpt`, 2690316669 B (2.51 GB), sha256 `9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1` |
| resolved TF32 | `matmul_tf32=False`, `cudnn_tf32=True`, `float32_matmul_precision=highest` — shipped default, untouched |
| cuEquivariance | absent from both venvs for arms A/B/D; installed into v_head last, for arm C only |

Fixture sha256, verified **on the box** against the worktree:

```
4c73e10fedc30eafc835744da7ab378258820657e3d4d13da142f17e34d39e4c  perf/dsfix/fixtures/rfd3_R4.json
647e066a983e66184e16bf7696b6e731f354e4161c6e764b292e1f9a15c00eef  perf/dsfix/fixtures/rfd3_R4_gpu.json
cecc4d30e2898fc3fcafd0152dcb1cb14aca18315244f2d88db968c4321dd90b  perf/dsfix/targets/R4_9q6y_A.pdb
```

## The install difference, proven not assumed

`gpu_rfd3_setup.sh` prints this per venv. Upstream has no `pyproject.toml` under `models/rfd3`, so
the editable install fails and the script's fallback overlays the `4010e3e2e` `attention.py` onto a
0.2.0 install. Route taken: **OVERLAY** (`/work/v_head/INSTALL_ROUTE`).

| symbol | v_head (arm A) | v_pip (arm B) |
|---|---|---|
| `dense_sdpa_pairbias_attention` | **True** | False |
| `use_dense_sdpa_pairbias` | **True** | False |
| `sparse_pairbias_attention` | True | True |

## Two things that cost a cycle here

- **`foundry install rfd3` is a separate step** and was missing from the first setup script. Without
  it the design dies in `foundry/inference_engines/base.py:86` with `Invalid checkpoint: rfd3. And
  could not find checkpoint in default installation location:
  /root/.foundry/checkpoints/rfd3_latest.ckpt`. The 4-timestep smoke caught it in 30 s, before any
  measured point — the counters read all-zero and the guard refused to continue, which is exactly
  what guarding on the output rather than the exit code is for. Download is ~7 min.
- **Detached remote launch** needs `(setsid nohup CMD </dev/null >/dev/null 2>&1 &)` — the whole
  thing in a subshell with all three fds redirected. Without the subshell, ssh holds the channel
  open waiting on the inherited fds and the local client hangs until timeout.

## Progress

- [x] H200 rented, idle power and power limit measured
- [x] fixtures shipped, sha256 verified on the box
- [x] arm A + arm B venvs installed, difference proven
- [x] checkpoint installed, sha256 recorded
- [ ] arm sequence: `bash /work/gpu_rfd3_runall.sh H200 700 78.7`, relaunched 14:41Z with weights
      present. Results append to `/work/results/rfd3_prod.jsonl`, log `/work/runall.log`, terminal
      marker `/work/RUN_OK` or `/work/RUN_FAIL`. Order: smoke, A b=8, A b=1, D, B, cueq install, C.
- [ ] pull the JSONL back into `perf/dsfix/results/`
- [ ] tear down H200, confirm zero instances
- [ ] B200 leg (offer 33945597, $5.3138/hr, Oregon)
- [ ] write `state/rfd3-gpu-h200-b200.md`
