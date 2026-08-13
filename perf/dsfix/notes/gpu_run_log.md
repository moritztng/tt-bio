# rfd3-gpu-h200-b200 — live rental log

Append-only. **A relaunch reads this FIRST** to find a box that is still burning money, and to avoid
re-paying for an install that already happened.

## Boxes — BOTH TORN DOWN, nothing of this task is still billing

| box | vast instance | offer | $/hr | rented (UTC) | destroyed (UTC) | cost |
|---|---|---|---:|---|---|---:|
| H200 | 47635549 | 39605030 | 3.9561 | 14:15 | **15:27, confirmed** | ~$4.74 |
| B200 | 47639947 | 47561350 | 5.3490 | 15:27 | **16:04, confirmed** | ~$3.35 |

Verified by a separate list read after each destroy: zero instances labelled `rfd3-gpu-*`. The box
`47636751 prot-odde-b200` seen in those reads belongs to another worker and was left alone.

Credit before renting: **$34.3722** (`vastai show user --raw`, agrees with the fixture doc's
post-teardown reading).

Two vast CLI gotchas that each cost rental time:

- `show instances-v1 --raw` returns a **dict** with an `instances` key, not a list. Indexing it as a
  list throws `TypeError: string indices must be integers` on every poll of a wait loop.
- `destroy instance` **prompts for confirmation** and aborts without it. Pipe `yes |`, then confirm
  with a separate list read.
- this CLI build has no `--price` flag on `create instance`; on-demand takes the offer price.

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

## Progress — COMPLETE

- [x] H200 rented, idle 78.7 W and 700 W limit measured
- [x] fixtures shipped, sha256 verified on both boxes
- [x] arm A + arm B venvs installed, difference proven by symbol presence
- [x] checkpoint installed, sha256 9b3f8592, identical on both boxes
- [x] H200: all 5 arms measured and valid
- [x] H200 torn down, confirmed
- [x] B200 rented, idle 147.9 W and 1000 W limit measured; all 5 arms measured and valid, no hang
- [x] B200 torn down, confirmed
- [x] results in `perf/dsfix/results/rfd3_prod_{h200,b200}.jsonl`
- [x] `state/rfd3-gpu-h200-b200.md` written, DONE_CHECK passes

The B200 leg needed no rediscovery: install plus checkpoint took under 2 minutes there, against ~15
on the H200, because every trap below was already fixed in the committed scripts.

## Correction, made before the deliverable was final

An earlier commit message claimed arm C's H200 median (104.93 s) sits inside arm A's warm range
"[103.57, 104.97]". That upper bound is arm A's **cold** batch, not a warm rep. Arm A's warm range is
103.57-103.87, so arm C is +1.10 % against it, not contained by it. On the B200 arm C is -0.15 %.
Both are nulls next to the 11.9 % the dense path moves on the H200, and the counters are identical, so
the conclusion is unchanged: cuequivariance does nothing here. `state/rfd3-gpu-h200-b200.md` states it
the correct way.
