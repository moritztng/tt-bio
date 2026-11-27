# RFD4 Device-Only Step Ceiling: Per-Term Table (D89)

**Scope: "where this goes", NOT "where it is". Do not read this as a G4 PASS.**

Protocol (matches PERF/SIZING): crop 256, batch 1 x exploration_k 4, depth 2 (`batch00.npz`), `attn_fp32=on` (D84), Muon+Adam over `178,448,304` trained params (282 Muon `175,994,112` + 276 Adam `2,454,192`, `MUON_EXCLUDE` from `tt_bio/train/rfd4_objective.py:343-344`), last-pass recycle grad.
Price basis (identical to `~/r4t/sizing/proj_v1.py` / G4 protocol): BH Galaxy `$160,000 / 32` (`$5,000/chip`) vs DGX B200 `$562,154 / 8` (`$70,269.25/GPU`), list-price ratio `14.05385x`; B200 depth-2 baseline `0.175 s/step` (`2.459424 / T_tt` per chip).

Measured provenance (`p1_def`): `host=qb2 chip=2 board=p300c` (`110` worker Tensix cores), `tt-metal-b2z` (`ttnn 0.68.0`, Tracy `DEVICE KERNEL DURATION [ns]`), `trunk_step --cand-levers serving --win-pair card --attn-fp32 on`, `sha=9c1bbe2ff` (code) / `2ca22525c` (analysis), AICLK sampled during: `1350 MHz` median (`[1306, 1350]` step 2, `[1300, 1350]` step 3; external 0.5 Hz log `n=41` min/med/max `1350/1350/1350`), `n=2` warm steps (steps 2-3 of 3, `0.07%` step-to-step variance), digests step 1 `e4eb1bbaf835a92e` / step 2 `b18b62baf3e65c06` / step 3 `9ae125835a006b19`. Raw CSV: `qb2:~/r4t/ceiling/p1_def/ops.csv` (38,286 rows, 33 signposts).

---

## 1. Blackhole Hardware Peaks (from `knowledgebase/knowledge/`)

| Hardware | Worker cores | LoFi TF/s | HiFi2 TF/s | HiFi4 TF/s | Split-bf16 fp32 TF/s | DRAM GB/s | HiFi4 ridge (FLOP/B) | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BH p300c (qb2 measured) | 110 | 608.3 | 304.1 | 152.1 | 50.7 | 512 | 297 | `knowledge/infra/machines.md:150`, `knowledge/marketing/claims.md:60-62`, `ops.csv` grid=110 |
| BH Galaxy (32-chip target) | 120 | 663.6 | 331.8 | 165.9 | 55.3 | 512 | 324 | `knowledge/marketing/claims.md:233` (120 Tensix/chip, 512 GB/s) |
| BH p150a (140 physical) | 140 | 774.1 | 387.1 | 193.5 | 64.5 | 512 | 378 | `knowledge/infra/machines.md:150` (140 Tensix, 774 FP8, 512 GB/s) |
| BH p300c measured ttnn roof | 110 | - | - | 115.69 | 38.56 | 440.0 | 263 | `state/of3t/BACKWARD.md:243` |
| BH p150a measured ttnn roof | 130 | 161.14 | 140.35 | 100.55 | 33.52 | 435.2 | 231 | `PLAYBOOKS.md:443`, `state/rfd4/ARCH.md:107-111` |

FPU derivation: `2048 MAC/cycle/Tensix` LoFi at 1.35 GHz (`774 FP8 TFLOPS / (140 x 1.35 GHz x 2)`), HiFi2 = 1/2 LoFi, HiFi4 = 1/4 LoFi, split-bf16 exact fp32 matmul (3 HiFi4 passes, D34) = 1/3 HiFi4. BH Galaxy has +9.1% more compute cores than qb2 p300c at the same 512 GB/s DRAM bandwidth, so qb2 p300c measurements are conservative by ~3-5% on compute-bound ops.

---

## 2. Analytic FLOP/Byte Count and Roofline (`p300c` 110c HiFi4 152.1 TF/s, 512 GB/s)

Exact aten graph count on `batch00.npz` (`perf/r4t_ceiling/roofline.py`, `out/count_batch00.json`, `out/iso_batch00.json`):

| Component | TFLOP (HiFi4-eq) | DRAM GB (unfused) | Compute roof ms | DRAM roof ms | Roof 1-chip ms | Roof ZeRO-1 (1/32) ms |
|---|---:|---:|---:|---:|---:|---:|
| `cand_k` (K=4, d=2, grad-off, incl. `attn_fp32` +5.46 GB) | 15.664 | 336.6 | 103.0 | 657.4 | 657.4 | 657.4 |
| `win_fwd` (K=1, d=2, taped, incl. `attn_fp32` +1.37 GB) | 3.915 | 81.6 | 25.7 | 159.4 | 159.4 | 159.4 |
| `win_bwd` (K=1, d=2, incl. `attn_fp32` +1.63 GB) | 3.914 | 86.3 | 25.7 | 168.5 | 168.5 | 168.5 |
| `fold_vjp` (`DeviceFolds` + fp32 grad accum, 529 params) | 0.000 | 2.14 | 0.0 | 4.18 | 4.18 | 4.18 |
| `split_rebuild` (`DeviceFolds` bf16 + row-scale + caches) | 0.000 | 1.53 | 0.0 | 2.99 | 2.99 | 2.99 |
| `opt_muon_ns` (282 mats, 5 quintic iters, split-bf16 fp32) | 8.398 | 16.64 | 55.22 | 32.50 | 55.22 | 1.73 |
| `opt_elementwise` (282 Muon + 276 Adam + EMA + clip, fp32) | 0.005 | 5.73 | 0.04 | 11.19 | 11.19 | 0.35 |
| **Fully-fused compute floor (model compute + system)** | **31.90** | — | — | — | **228.1 ms (10.78x/$)** | **163.7 ms (15.02x/$)** |
| **Unfused-aten DRAM roof (zero fusion + system)** | **31.90** | **530.5** | — | — | **1058.8 ms (2.32x/$)** | **994.5 ms (2.47x/$)** |

Isolated host-side submodules (already inside model totals above, priced separately in Section 4 so nothing is double-counted):
- `pair_const_c1` (C1 proj + K=4 norm scales): `53.15 GFLOP`, `9.30 GB` -> `18.16 ms` roof.
- `glue_heads` (fwd K+1 + bwd): `5.43 GFLOP`, `0.25 GB` -> `0.48 ms` roof.
- `glue_cond_init` (fwd K+1 x d + bwd): `11.69 GFLOP`, `0.52 GB` -> `1.01 ms` roof.

---

## 3. Measured Device Kernel Time & Achieved Roofline (`p1_def`, warm mean steps 2-3)

Compute phases (`fwd` + `cand_pair` + `cand_trunk` + `win_pair` + `win_trunk` + `tape_bwd`; excludes host-IO `fold_bwd` 14.69 ms and `rebuild` 4.09 ms). Measured total matmul FLOPs = `23.28 TF` vs analytic `23.41 TF` (`99.4%` match). All matmuls ran `HiFi4`.

| Op class | Meas ms | % card | DRAM GB | TFLOP | Ach GB/s | % 512 GB/s | Ach TF/s | % 152.1 TF/s | Per-op roof ms | Roof/meas | Ops/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `move` | 638.72 | 34.6% | 123.7 | 0.00 | 193.6 | 37.8% | 0.0 | 0.0% | 241.55 | 37.8% | 3380 |
| `matmul` | 585.93 | 31.7% | 117.4 | 23.28 | 200.3 | 39.1% | 39.7 | 26.1% | 229.88 | 39.2% | 1933 |
| `eltwise` | 412.90 | 22.4% | 156.9 | 0.00 | 380.1 | 74.2% | 0.0 | 0.0% | 306.51 | 74.2% | 3735 |
| `norm` | 126.65 | 6.9% | 40.1 | 0.00 | 316.3 | 61.8% | 0.0 | 0.0% | 78.25 | 61.8% | 502 |
| `generic_fused` | 53.01 | 2.9% | 19.3 | 0.00 | 363.5 | 71.0% | 0.0 | 0.0% | 37.64 | 71.0% | 108 |
| `reduce` | 25.94 | 1.4% | 3.8 | 0.00 | 146.4 | 28.6% | 0.0 | 0.0% | 7.42 | 28.6% | 430 |
| `softmax` | 3.42 | 0.2% | 0.9 | 0.00 | 257.9 | 50.4% | 0.0 | 0.0% | 1.72 | 50.4% | 56 |
| **TOTAL** | **1846.56** | **100.0%** | **462.0** | **23.28** | **250.2** | **48.9%** | **12.6** | **8.3%** | **902.96** | **48.9%** | **10144** |

Measured efficiencies used to price unported terms in Section 4:
- Matmul (`matmul` class): `39.2%` of per-op roofline (`MinimalMatmul` = `61.6 TF/s` / `48.0%` roof; `tape_bwd` `MatmulDeviceOperation` = `21.2 TF/s` / `21.6%` roof).
- Streaming DRAM (`BinaryNg`/`Typecast`/`Concat`/`Unary`): `382.6 GB/s` = `74.7%` of 512 GB/s (`87.0%` of 440 GB/s measured roof).
- Mixed non-matmul DRAM (incl. `Permute`/`Transpose`/`LayerNorm`/`Reduce`): `269.4 GB/s` = `52.6%` of 512 GB/s.

Firmware/dispatch observation: across `10,144` unfused compute ops/step, `DEVICE FW DURATION` = `2536.28 ms` (`+689.72 ms` over `1846.56 ms` kernel time; actual wall `fwd+bwd` minus ~220 ms host glue is `~2300 ms`, i.e. ~`+450 ms` real FW/dispatch overhead on 10,144 ops). Cutting op count ~3x via K14 + fusion cuts this dispatch overhead proportionally.

---

## 4. Per-Term Device-Only Ceiling (Default Card Kernels, Directly Measured)

| Term | ZeRO-1 (1/32) ms | 1-chip (1/1) ms | % ZeRO-1 | Basis |
|---|---:|---:|---:|---|
| `cand_pair` (`PairOnCard` K=4) | 9.55 | 9.55 | 0.5% | measured device (`p1_def`) |
| `cand_trunk` (K=4, d=2, grad-off, default unfused) | 1062.42 | 1062.42 | 56.1% | measured device (`p1_def`) |
| `win_pair` (P2 `PairOnCard.taped` fwd) | 2.35 | 2.35 | 0.1% | measured device (`p1_def`) |
| `win_trunk` (K=1, d=2: cyc0 `172.84` + cyc1 `170.05`) | 342.98 | 342.98 | 18.1% | measured device (`p1_def`) |
| `tape_bwd` (trunk + P2 pair bwd) | 429.11 | 429.11 | 22.6% | measured device (`p1_def`) |
| `fwd` misc (masks) | 0.15 | 0.15 | 0.0% | measured device (`p1_def`) |
| `pair_const_c1` (C1 proj + K=4 norm scales) | 26.54 | 26.54 | 1.4% | roof / measured class eff |
| `glue_heads` (fwd K+1 + bwd) | 1.03 | 1.03 | 0.1% | roof / measured class eff |
| `glue_cond_init` (fwd K+1 x d + bwd) | 2.15 | 2.15 | 0.1% | roof / measured class eff |
| `fold_vjp` (`DeviceFolds` + fp32 grad accum) | 7.95 | 7.95 | 0.4% | roof / measured class eff |
| `split_rebuild` (`DeviceFolds` bf16 + caches) | 5.69 | 5.69 | 0.3% | roof / measured class eff |
| `opt_muon_ns` (282 mats, 5 iters, split-bf16 fp32) | 4.40 | 140.76 | 0.2% | roof / measured class eff |
| `opt_elementwise` (Muon+Adam+EMA+clip, fp32) | 0.47 | 14.98 | 0.0% | roof / measured class eff |
| Host batch feed | 0.00 | 0.00 | 0.0% | overlapped |
| **CEILING (default card kernels, host overlapped)** | **1894.8 ms (1.895 s)** | **2045.6 ms (2.046 s)** | **100.0%** | |

- **Default card kernels (ZeRO-1 1/32)**: **`1.8948 s/step`** -> **`1.30x/$`** per chip vs B200 (`1.17x` at 0.90 DP eff, `1.10x` at 0.85 DP eff; 8 GLX scratch `26.9 d`, `$31k` vs B200 `$37k`). Unsharded 1-chip: `2.0456 s` -> `1.20x/$`.
- **Conclusion**: removing all host glue alone (`5.24 s -> 1.895 s`, `0.47x -> 1.30x`) does NOT clear G4 `1.50x/$` (`<= 1.64 s` per chip, `<= 1.48 s` at 0.90 DP, `<= 1.39 s` at 0.85 DP). Card kernel fusion is mandatory alongside device residency.

---

## 5. Gap-to-Roofline Ladder (Where the Step Goes Once Card Kernels Match Serving/Roofline)

| Stage (device-only, ZeRO-1 1/32, host overlapped) | Step s | Per-chip x/$ | x/$ @ 0.90 DP | x/$ @ 0.85 DP | 8 GLX scratch (eff 0.90) | Basis |
|---|---:|---:|---:|---:|---:|---|
| 0. Today's host-bound step (PERF, K13) | 4.820 | 0.51x | 0.46x | 0.43x | 271 d (actual 0.325 eff) | measured wall (`rows/perf.md`) |
| **1. Device-resident, default card kernels (`p1_def`)** | **1.895** | **1.30x** | **1.17x** | **1.10x** | **26.9 d ($31k)** | **measured device (`p1_def`) + Section 4** |
| **2. + K14 (`cand_trunk` fused serving kernels, `-784 ms`)** | **1.111** | **2.21x** | **1.99x** | **1.88x** | **15.8 d ($18k)** | **measured `e2e-256-b4-v10.json` (`264.2 ms` @ 1350 MHz) + `14.3 ms` `attn_fp32`** |
| 3. + `win_trunk` cyc0 grad-off fused (`-128 ms`) | 0.983 | 2.50x | 2.25x | 2.13x | 13.9 d ($16k) | measured `e2e-256-b1-v10.json` (`42.9 ms` @ 1343 MHz) + `1.8 ms` `attn_fp32` |
| 4. + `tape_bwd` matmuls (`dW` split-K + `dX` `MinimalMatmul`, `-107 ms`) | 0.876 | 2.81x | 2.53x | 2.39x | 12.4 d ($15k) | `p1_def` shape audit below (`6.8-35 TF/s` -> `61-67 TF/s`) |
| 5. Analytic compute floor (fully fused + ZeRO-1) | 0.164 | 15.02x | 13.52x | 12.77x | 2.3 d ($3k) | Section 2 HiFi4 peak |

*(Unsharded 1-chip adds `+150.9 ms`: Stage 2 `1.262 s` = `1.95x/$`, Stage 4 `1.027 s` = `2.40x/$`.)*

### Largest Measured Gaps to Roofline (Exact Offenders from `p1_def/ops.csv`)
1. **`cand_trunk` unfused (`1062.42 ms` -> `~278 ms`, `-784 ms`)**: `0` `GenericOp` calls ran. `315.0 ms` is literally 112 triangle-mult layout ops (`Permute (4,256,256,256) {0,3,1,2}` 56 ops `188.2 ms`, `Transpose HC` 28 ops `107.3 ms`, `Transpose WH` 28 ops `19.6 ms`), plus `232.3 ms` `BinaryNg`, `100.5 ms` `LayerNorm`, `95.0 ms` `Slice`/other moves. K14 (`f79709926`) unlocks all 28 serving fused kernels already proven at `264.2 ms` on the exact same qb2 p300c board (`state/rfd4/census/e2e-256-b4-v10.json`, AICLK 1350 MHz).
2. **`win_trunk` cycle 0 grad-off (`172.84 ms` -> `~44.7 ms`, `-128 ms`)**: `win_trunk` step 3 splits `172.84 ms` (cycle 0, grad-off) + `170.05 ms` (cycle 1, taped). Cycle 0 needs no tape and can use K14's `ag.suspended()` + serving fused kernels (`e2e-256-b1-v10.json` first pass = `42.89 ms` + `1.8 ms` `attn_fp32`).
3. **`tape_bwd` `MatmulDeviceOperation` (`184.26 ms` @ `21.2 TF/s` = `21.6%` roof, `-107 ms`)**:
   - **Weight-gradient `dW` (`K=65536`, small `M x N`, `bf16->fp32`)**: `256x384` (14 ops) takes `26.45 ms` (`6.8 TF/s`, 4.5% peak), `384x384` (15 ops) takes `23.43 ms` (`12.4 TF/s`), `384x1024` (14 ops) takes `22.35 ms` (`32.3 TF/s`), `384x768` (7 ops) takes `11.17 ms` (`24.2 TF/s`) — total `83.4 ms` on shapes that do not saturate 110 cores while `K=2048` tiles serialize! By contrast `768x384` (21 ops, 288 output tiles) hits `67.0 TF/s` (`12.11 ms`). Fixing blocking / split-K on `dW` recovers `~62 ms`.
   - **Activation-gradient `dX` (`M=65536`, `bf16->bf16`)**: `~60+ ms` runs on `MatmulDeviceOperation` at `29-35 TF/s` vs `MinimalMatmul` at `61.6 TF/s` (L4 already proved `-17.8%` on PairUpdate bwd; recovers `~45 ms`).
