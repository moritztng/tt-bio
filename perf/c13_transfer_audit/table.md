| # | slug | lever | screen kind | predicted | measured | unit | level | R | outcome | in band | source |
|--:|---|---|---|--:|--:|---|---|--:|---|---|---|
| 1 | `b2z-sampler-steps` | 200 -> 50 sampling steps on BH* *(unscored)* | count/rate model | +4.8780 | +3.9600 | s | fold | 0.812 | HELD | -- | `b2z-sampler-steps.md:13-23` |
| 2 | `b2z2-dual-chip-fold` | sharded dual-chip fold | count/rate model | +2.1900 | -0.0317 | s | fold | -0.014 | FLIPPED | -- | `b2z2-dual-chip-fold.md:242-245,393-394` |
| 3 | `b2z2-sharded-sampler` | token-axis sampler shard (P3) | count/rate model | 1.15000x | 0.99300x | ratio-1 | fold | -0.047 | FLIPPED | **no** | `b2z2-sharded-sampler.md:22-35` |
| 4 | `c10-core-grid` | core_grid on bare sites (own prereg) | count/rate model | +0.2000 | +0.1446 | s | fold | 0.723 | HELD | yes | `c10-core-grid.md:11-16` |
| 5 | `c10-trace-lever` | DiT trace replay | count/rate model | +3.0400 | -0.0214 | s | fold | -0.007 | FLIPPED | **no** | `c10-trace-lever.md:21-29,37` |
| 6 | `c12-host-decomp` | F = 3.9830 s read as deletable host | count/rate model | +3.9830 | +0.0960 | s | fold | 0.024 | HELD | -- | `c12-host-decomp.md:492-498,1129-1134` |
| 7 | `b2z2-bh-compose-landed` | K2 on BH | fold ratio transferred | 1.02000x | 1.00448x | ratio-1 | fold | 0.224 | HELD | **no** | `b2z2-bh-compose-landed.md:25-38` |
| 8 | `b2z2-bh-compose-landed` | DST-resident on BH *(unscored)* | fold ratio transferred | 1.00200x | 0.99748x | ratio-1 | fold | -1.260 | FLIPPED | -- | `b2z2-bh-compose-landed.md:25-38` |
| 9 | `b2z2-bh-compose-landed` | MSA ladder on BH | fold ratio transferred | 1.04000x | 1.04650x | ratio-1 | fold | 1.162 | HELD | **no** | `b2z2-bh-compose-landed.md:25-38` |
| 10 | `b2z2-bh-compose-landed` | K2+DST+MSA union on BH | fold ratio transferred | 1.06000x | 1.05985x | ratio-1 | fold | 0.998 | HELD | yes | `b2z2-bh-compose-landed.md:25-38` |
| 11 | `b2z2-bh-stack-atom` | corrected five-lever stack on BH | fold ratio transferred | 1.12040x | 1.11982x | ratio-1 | fold | 0.995 | HELD | -- | `b2z2-bh-stack-atom.md:88-96` |
| 12 | `b2z2-bh-union-clean` | UNION of three on BH | fold ratio transferred | 1.07000x | 1.09858x | ratio-1 | fold | 1.408 | HELD | **no** | `b2z2-bh-union-clean.md:215-233` |
| 13 | `b2z2-bh-union-clean` | HOST on BH | fold ratio transferred | 1.04500x | 1.04642x | ratio-1 | fold | 1.032 | HELD | yes | `b2z2-bh-union-clean.md:215-233` |
| 14 | `b2z2-bh-union-clean` | SILU on BH | fold ratio transferred | 1.02300x | 1.02423x | ratio-1 | fold | 1.053 | HELD | yes | `b2z2-bh-union-clean.md:215-233` |
| 15 | `b2z2-bh-union-clean` | AKW on BH | fold ratio transferred | 1.01000x | 1.02744x | ratio-1 | fold | 2.744 | HELD | **no** | `b2z2-bh-union-clean.md:215-233` |
| 16 | `b2z2-bh-union-step` | five-lever STACK on BH | fold ratio transferred | 1.11480x | 1.12862x | ratio-1 | fold | 1.120 | HELD | **no** | `b2z2-bh-union-step.md:63-66` |
| 17 | `b2z2-conf-device-ship` | confidence head device port on BH | fold ratio transferred | 1.04000x | 1.04743x | ratio-1 | fold | 1.186 | HELD | yes | `b2z2-conf-device-ship.md:379-383` |
| 18 | `b2z2-elision-bh-measure` | TT_BIO_ATOM_SHIFT_GATHER on BH | fold ratio transferred | 1.00800x | 1.01985x | ratio-1 | fold | 2.481 | HELD | **no** | `b2z2-elision-bh-measure.md:20-31` |
| 19 | `b2z2-host-device-compose` | union of three host->device ports, WH | fold ratio transferred | 1.05200x | 1.06495x | ratio-1 | fold | 1.249 | HELD | **no** | `b2z2-host-device-compose.md:33-49` |
| 20 | `b2z2-trunk-bh-leg` | all3 trunk byte levers on BH (P1) | fold ratio transferred | 1.02000x | 1.01522x | ratio-1 | fold | 0.761 | HELD | yes | `b2z2-trunk-bh-leg.md:30-38` |
| 21 | `b2z2-trunk-bh-leg` | qkvg alone on BH (P2) *(unscored)* | fold ratio transferred | 1.00100x | 1.00748x | ratio-1 | fold | 7.480 | HELD | **no** | `b2z2-trunk-bh-leg.md:32-40` |
| 22 | `b2z2-trunk-byte-round2-ship` | 3 trunk byte flags, post-merge re-read | fold ratio transferred | 1.01522x | 1.01573x | ratio-1 | fold | 1.034 | HELD | -- | `b2z2-trunk-byte-round2-ship.md:19-24` |
| 23 | `b2z2-zinit-ship` | TT_BIO_DEVICE_ZINIT on BH | fold ratio transferred | 1.01090x | 1.01955x | ratio-1 | fold | 1.794 | HELD | -- | `b2z2-zinit-ship.md:1208-1212` |
| 24 | `c12-reblock-delete` | fused gated in-projection | novel mechanism | +1.0062 | -2.7578 | s | fold(op x 560 calls) | -2.741 | FLIPPED | **no** | `c12-reblock-delete.md:786,808,880` |
| 25 | `b2z2-atom-shard-wire` | atom-axis window shard (P1) | op/block A/B | 1.13000x | 0.99380x | ratio-1 | fold | -0.048 | FLIPPED | **no** | `b2z2-atom-shard-wire.md:22-24` |
| 26 | `b2z2-everything-union-wh` | whole union on WH (P1) | op/block A/B | 1.08500x | 1.10789x | ratio-1 | fold | 1.269 | HELD | yes | `b2z2-everything-union-wh.md:'## 7' P1` |
| 27 | `b2z2-qchunk-isolated-bh` | SDPA q-chunk BH fold (P4) | op/block A/B | 1.00330x | 1.00355x | ratio-1 | fold | 1.076 | HELD | yes | `b2z2-qchunk-isolated-bh.md:34-37` |
| 28 | `b2z2-trunk-fold-ab-bh` | all3 trunk byte levers, WH fold (P1) | op/block A/B | 1.02450x | 1.02648x | ratio-1 | fold | 1.081 | HELD | yes | `b2z2-trunk-fold-ab-bh.md:28-34` |
| 29 | `b2z2-trunk-fold-ab-bh` | qkvg alone, WH fold (P2) | op/block A/B | 1.00730x | 1.00213x | ratio-1 | fold | 0.292 | HELD | **no** | `b2z2-trunk-fold-ab-bh.md:29,34-36` |
| 30 | `c10-qchunk-sign` | SDPA q-chunk rule, 512 aa | op/block A/B | +0.0450 | +0.0643 | s | fold | 1.429 | HELD | -- | `c10-qchunk-sign.md:29,47` |
| 31 | `c10-qchunk-sign` | SDPA q-chunk rule, 298 aa | op/block A/B | +0.0300 | +0.8288 | s | fold | 27.627 | HELD | **no** | `c10-qchunk-sign.md:31-32,48` |
| 32 | `c12-compose-fold` | TT_BIO_DIT_COND_HOIST | op/block A/B | +0.2134 | +0.2880 | s | fold | 1.350 | HELD | -- | `c12-compose-fold.md:918-922; c12-cond-hoist-block-timing.md:208` |
| 33 | `c12-compose-fold` | TT_BIO_UNFUSED_SILU | op/block A/B | +0.2843 | +0.3342 | s | fold | 1.176 | HELD | -- | `c12-compose-fold.md:918-922; c12-unfused-silu-bh.md:74-82` |
| 34 | `c12-compose-fold` | hoist+silu stack | op/block A/B | +0.4977 | +0.5756 | s | fold | 1.157 | HELD | -- | `c12-compose-fold.md:918-920,928-934` |
| 35 | `c10-core-grid` | core_grid on bare sites (inherited screen) | replay gap | +4.9080 | +0.1446 | s | fold | 0.029 | HELD | -- | `c10-core-grid.md:13-17,57-60,134` |
| 36 | `b2z2-orchestrator` | wave-2 campaign headline | roof headroom | 1.60000x | 1.00000x | ratio-1 | fold | 0.000 | NULLED | **no** | `b2z2-orchestrator.md:18-31` |
| 37 | `b2z2-everything-union-wh` | sampler stage (P6) | op/block A/B | 1.11000x | 1.18816x | ratio-1 | fold-stage | 1.711 | HELD | **no** | `b2z2-everything-union-wh.md:'## 7' P6` |
| 38 | `b2z2-everything-union-wh` | trunk stage (P6) | op/block A/B | 1.02000x | 1.04429x | ratio-1 | fold-stage | 2.215 | HELD | **no** | `b2z2-everything-union-wh.md:'## 7' P6` |
| 39 | `b2z2-everything-union-wh` | confidence stage (P6) *(unscored)* | op/block A/B | 1.15000x | 1.58106x | ratio-1 | fold-stage | 3.874 | HELD | yes | `b2z2-everything-union-wh.md:'## 7' P6` |
| 40 | `b2z2-everything-union-wh` | conditioning stage (P6) *(unscored)* | op/block A/B | 1.30000x | 2.97452x | ratio-1 | fold-stage | 6.582 | HELD | yes | `b2z2-everything-union-wh.md:'## 7' P6` |
| 41 | `c10-orchestrator` | arithmetic_free_traffic bracket | roof headroom | +2.2985 | +4.1310 | s | measured | 1.797 | HELD | **no** | `c10-orchestrator.md:692` |
| 42 | `c12-matmul-key-attribution` | matmul|1x128x512x512|K=512 prize | roof headroom | +0.7794 | +0.0584 | s | in-situ fold s | 0.075 | HELD | -- | `c12-matmul-key-attribution.md:233` |
| 43 | `c12-matmul-key-attribution` | whole `matmul` class prize | roof headroom | +0.9200 | +0.1352 | s | in-situ fold s | 0.147 | HELD | -- | `c12-matmul-key-attribution.md:234` |
| 44 | `c12-matmul-key-attribution` | matmul|out=1024x32x512|K=512 prize | roof headroom | +0.1067 | +0.0000 | s | in-situ fold s | 0.000 | NULLED | -- | `c12-matmul-key-attribution.md:246-250` |
| 45 | `b2z-diffusion-utilization` | matmul grid height on a 16-tile row axis | roof headroom | +0.5042 | +0.0054 | s | arith re-derive | 0.011 | HELD | -- | `b2z2-sampler-ceiling-map.md:14-15,115` |
| 46 | `b2z2-step-binaryng-fusion` | 2-chain BinaryNg fusion (P2) | count/rate model | 1.04350x | 0.98414x | ratio-1 | step | -0.365 | FLIPPED | **no** | `b2z2-step-binaryng-fusion.md:31-37,49-53` |
| 47 | `b2z2-step-fusion-next-sites` | unpadded head split (P2) | count/rate model | 0.98000x | 0.73670x | ratio-1 | step | 13.165 | HELD | **no** | `b2z2-step-fusion-next-sites.md:50-54` |
| 48 | `b2z2-step-layernorm-fusion` | AdaLN shared s-norm | count/rate model | 1.03900x | 1.03932x | ratio-1 | step | 1.008 | HELD | yes | `b2z2-step-layernorm-fusion.md:21-23` |
| 49 | `b2z2-step-layout-elision` | arm A padded tail | count/rate model | 1.01730x | 1.02549x | ratio-1 | step | 1.473 | HELD | -- | `b2z2-step-layout-elision.md:25-34` |
| 50 | `b2z2-step-layout-elision` | arm B fused qkv head split | count/rate model | 1.06230x | 1.04124x | ratio-1 | step | 0.662 | HELD | -- | `b2z2-step-layout-elision.md:28-35` |
| 51 | `b2z2-step-matmul-group` | W1 s-norm fold | count/rate model | 1.03540x | 1.03975x | ratio-1 | step | 1.123 | HELD | -- | `b2z2-step-matmul-group.md:49-51` |
| 52 | `b2z2-atom-window-next` | K/V projection onto the atom axis | op/block A/B | 0.5820 | 0.5967 | ms/step | step | 1.025 | HELD | -- | `b2z2-atom-window-next.md:22-27` |
| 53 | `b2z2-step-fusion-next-sites` | atom L1 residency (P1) | op/block A/B | 1.03400x | 1.08461x | ratio-1 | step | 2.489 | HELD | **no** | `b2z2-step-fusion-next-sites.md:42-45` |
| 54 | `b2z2-step-program-fusion` | indexing-matrix window build | op/block A/B | 3.9700 | 2.8800 | ms/step | step | 0.725 | HELD | -- | `b2z2-step-program-fusion.md:38-50` |
| 55 | `b2z-work-removal` | MSA depth ladder 1024 -> 64 | count/rate model | 1.67500x | 1.42200x | ratio-1 | layer | 0.625 | HELD | **no** | `b2z-work-removal.md:51-53` |
| 56 | `b2z2-msa-movement-attack` | L1-resident MSA row block (P3) | count/rate model | 1.05500x | 1.02603x | ratio-1 | layer | 0.473 | HELD | **no** | `b2z2-msa-movement-attack.md:34-36` |
| 57 | `b2z2-shard-replication-attack` | the b role's share of the constant (P2) | count/rate model | 16.0000 | 10.1800 | ms | block | 0.636 | HELD | -- | `b2z2-shard-replication-attack.md:44-48` |
| 58 | `b2z2-shard-replication-attack` | triatt_end's share of the constant (P2) | count/rate model | 3.0000 | 6.3460 | ms | block | 2.115 | HELD | -- | `b2z2-shard-replication-attack.md:44-46` |
| 59 | `b2z2-trunk-byte-floor` | triangle-attention normed-pair read (P2) | count/rate model | 1.02000x | 1.01460x | ratio-1 | block | 0.730 | HELD | yes | `b2z2-trunk-byte-floor.md:24-30` |
| 60 | `b2z2-trunk-byte-round2` | rank 2 site (P1) | count/rate model | 1.01460x | 1.01080x | ratio-1 | block | 0.740 | HELD | yes | `b2z2-trunk-byte-round2.md:24-33` |
| 61 | `b2z2-trunk-byte-round2` | rank 1b site (P2) | count/rate model | 1.00800x | 1.02491x | ratio-1 | block | 3.114 | HELD | **no** | `b2z2-trunk-byte-round2.md:26-33` |
| 62 | `b2z2-trunk-shard-scale-wh` | row shard block ratio @2 chips (P1) | count/rate model | 1.35000x | 1.34220x | ratio-1 | block | 0.978 | HELD | -- | `b2z2-trunk-shard-scale-wh.md:35-47` |
| 63 | `b2z2-trunk-shard-scale-wh` | row shard block ratio @4 chips (P1) | count/rate model | 1.70000x | 1.64040x | ratio-1 | block | 0.915 | HELD | -- | `b2z2-trunk-shard-scale-wh.md:35-47` |
| 64 | `b2z2-trunk-shard-scale-wh` | row shard block ratio @8 chips (P1) | count/rate model | 1.90000x | 1.94240x | ratio-1 | block | 1.047 | HELD | -- | `b2z2-trunk-shard-scale-wh.md:35-47` |
| 65 | `b2z2-mcast-operand-build` | multicast the shared operand | novel mechanism | 1.03500x | 0.93837x | ratio-1 | block | -1.761 | FLIPPED | **no** | `b2z2-mcast-operand-build.md:15-23` |
| 66 | `b2z2-datum-rate-floor` | bare move-only kernel ns/tile | count/rate model | 52.5000 | 30.4400 | ns/tile | op | 0.580 | HELD | **no** | `b2z2-datum-rate-floor.md:11-18` |
| 67 | `b2z2-dst-resident-fusion` | delete the mul_cb L1 round trip | count/rate model | 1.00800x | 1.05410x | ratio-1 | op | 6.763 | HELD | -- | `b2z2-dst-resident-fusion.md:10-16` |
| 68 | `b2z2-l1-sharded-residency` | L1LOCAL / DRAM per-tile cost | count/rate model | 0.875 | 0.343 | ratio | op | 0.392 | HELD | **no** | `b2z2-l1-sharded-residency.md:11-18` |
| 69 | `b2z2-shard-replication-attack` | b-shard collective cost @2 (P3) | count/rate model | 1.5000 | 1.7000 | ms | op | 1.133 | HELD | -- | `b2z2-shard-replication-attack.md:33-48` |
| 70 | `b2z2-shard-replication-attack` | b-shard collective cost @4 (P3) | count/rate model | 2.0200 | 1.9700 | ms | op | 0.975 | HELD | -- | `b2z2-shard-replication-attack.md:33-48` |
| 71 | `b2z2-step-adaln-sdpa` | norm-of-a parallelism (P2) | count/rate model | 1.01250x | 1.00000x | ratio-1 | op | 0.000 | NULLED | **no** | `b2z2-step-adaln-sdpa.md:28-36` |
| 72 | `b2z2-step-matmul-group` | N-stacking the s-projections *(unscored)* | count/rate model | 1.00250x | 0.83160x | ratio-1 | op | -67.360 | FLIPPED | **no** | `b2z2-step-matmul-group.md:54` |
| 73 | `b2z2-trunk-shard-scale-wh` | per-block link cost @2 chips (P5) | count/rate model | 6.2000 | 6.0000 | ms | op | 0.968 | HELD | -- | `b2z2-trunk-shard-scale-wh.md:39-42` |
| 74 | `b2z2-trunk-shard-scale-wh` | per-block link cost @4 chips (P5) | count/rate model | 9.3000 | 8.1000 | ms | op | 0.871 | HELD | -- | `b2z2-trunk-shard-scale-wh.md:39-42` |
| 75 | `b2z2-trunk-shard-scale-wh` | per-block link cost @8 chips (P5) | count/rate model | 10.8000 | 8.6000 | ms | op | 0.796 | HELD | -- | `b2z2-trunk-shard-scale-wh.md:39-42` |
| 76 | `b2z2-twopass-loop-bh` | hoist the two-pass loop init (P2) | count/rate model | 1.03000x | 0.99750x | ratio-1 | op | -0.083 | FLIPPED | -- | `b2z2-twopass-loop-bh.md:21-26` |
| 77 | `c10-grid-sweep` | host issue per matmul call | count/rate model | 30.0000 | 6.9800 | us/call | op | 0.233 | HELD | **no** | `c10-grid-sweep.md:10-27` |
| 78 | `c12-fused-eltwise-at-pin` | gate sigmoid into the matmul epilogue | count/rate model | 8.1800 | 7.8400 | us/call | op | 0.958 | HELD | -- | `c12-fused-eltwise-at-pin.md:57-80` |
| 79 | `b2z2-fusion-rebuild` | fused trimul kernel on BH | fold ratio transferred | 1.02000x | 0.97400x | ratio-1 | op | -1.300 | FLIPPED | **no** | `b2z2-fusion-rebuild.md:21-24` |
| 80 | `b2z2-grid-qchunk-unify` | BH step ratio for the q-chunk rule (P1) | fold ratio transferred | 1.03250x | 1.13980x | ratio-1 | op | 4.302 | HELD | **no** | `b2z2-grid-qchunk-unify.md:22-34` |
| 81 | `b2z2-arrival-skew-attack` | two-injector skew absorber | novel mechanism | 1.12000x | 0.89157x | ratio-1 | op | -0.904 | FLIPPED | **no** | `b2z2-arrival-skew-attack.md:18-28` |
| 82 | `b2z2-pairformer-megakernel-build` | fused Pairformer trimul megakernel | novel mechanism | 0.9000 | -1.0255 | ms/call | op | -1.139 | FLIPPED | -- | `b2z2-pairformer-megakernel-build.md:44-46` |
| 83 | `roof-fuse-trimul-out` | g_out -> multiply_ epilogue, L1 product | op/block A/B | 1.05000x | 0.85120x | ratio-1 | op | -2.976 | FLIPPED | **no** | `roof-fuse-trimul-out.md:38-60` |
| 84 | `b2z-custom-sdpa` | exp as the missing SDPA term | roof headroom | 1.42500x | 1.04100x | ratio-1 | op | 0.096 | HELD | **no** | `b2z-custom-sdpa.md:103-106` |
| 85 | `b2z2-step-adaln-sdpa` | SDPA bias dtype (P4) | roof headroom | 1.02500x | 1.00000x | ratio-1 | op | 0.000 | NULLED | **no** | `b2z2-step-adaln-sdpa.md:26-38` |
| 86 | `b2z2-step-matmul-group` | W2 batched -> 2D matmul | roof headroom | 1.70000x | 1.01270x | ratio-1 | op | 0.018 | HELD | **no** | `b2z2-step-matmul-group.md:52-53` |
| 87 | `b2z2-shard-replication-attack` | fold route with the b-shard (P4) | count/rate model | 1.51000x | 1.46650x | ratio-1 | analytic | 0.915 | HELD | -- | `b2z2-shard-replication-attack.md:33-48` |
