# Protenix-v2 confidence head: device-resident port

The Protenix-v2 confidence head (z-embed + 4-block Pairformer + pae/pde/plddt
heads) runs the Pairformer on device but builds `z` on host, uploads the full
`(N,N,256)` pair tensor to the device Pairformer every sample, downloads
`(s_single, zf)`, and runs the heads on host — a per-sample `(N,N,256)`
device↔host round-trip. This port adds a device-resident path that keeps
`z_base = z_trunk + s1(s_inputs)[:,None] + s2(s_inputs)[None,:]` (the
sample-invariant part of `z`) resident on device across samples, so only the
per-sample `(N,3)` coordinates are uploaded and the pae/pde/plddt heads run on
device; only the small final logits are downloaded.

It is gated behind `TT_PROTENIX_CONF_DEVICE=1` **and** `NT >= 128`, and is **off
by default**. The default host-heads path is unchanged.

## Why it is off by default

The device path trades the per-sample `(N,N,256)` round-trip for a bf16
precision delta: the host path computes the full `z` in fp32 then rounds to bf16
once before the Pairformer, while the device path rounds `z_base` (once, resident)
and the per-sample distance-embed separately and adds them in bf16. At small N
this delta is amplified by the Pairformer into the plddt head, which is a 50-bin
expected-value (peak-position-sensitive). At the only real cached target
(`NT=38`) the device path regresses plddt; at large N it is parity-clean and a
real wall-clock win. The host-heads path is kept at small N (confidence is only
~23 ms at `NT=38`, ~0.5% of e2e, so there is nothing to win there).

The clean large-N PCC below is measured on padded repeat-block stress, **not** a
real large-N target — no real large-N target is cached on this host. Enabling the
device path by default at large N should wait for a real-large-N parity gate.

## Measured split (this host, BH 'pc', warm, device-synchronized)

Confidence-head internal timing, host-heads path (ms):

| N | z-embed host | upload z | Pairformer device | download | heads host | total | host fraction |
|---|---|---|---|---|---|---|---|
| 38 (real target) | 0.6 | 1.2 | 15.4 | 0.6 | 2.7 | 23.9 | 5.1 (22%) |
| 128 (padded) | 11.0 | 9.2 | 35.7 | 3.4 | 19.7 | 72.5 | 43.4 (60%) |
| 256 (padded) | 47.6 | 40.0 | 116.3 | 17.1 | 84.6 | 354.8 | 189.6 (53%) |

The device Pairformer at `N=256` is 116 ms — identical to the earlier qb1 BH
measurement (memory `protenix-accel-ceiling`), a clean cross-check. The host
fraction at `N=256` is 190 ms here vs 370 ms on qb1 (this host's CPU is faster);
the host-bound lever at large N holds.

## PCC (device path vs host-heads path)

| N | pae | pde | plddt |
|---|---|---|---|
| 38 (real target) | 0.94 | 0.97 | **0.29 (regressed)** |
| 128 (padded) | 0.978 | 0.986 | 0.994 |
| 256 (padded) | 0.987 | 0.993 | 0.997 |

The host-heads path itself is pae/pde PCC 1.0 and plddt PCC ~0.93 vs the real v2
reference (existing, `scripts/protenix_confidence_parity.py`). The device path
vs the host-heads path is the relevant comparison for the port. plddt at `NT=38`
regresses because the Pairformer input `z` diverges (`s_single` device-vs-host
PCC 0.71 at `NT=38`); the plddt logits stay close (PCC 0.95) but the 50-bin
softmax expected value is peak-position-sensitive, so a small logit shift
decorrelates the per-atom plddt. At `N>=128` the signal is large enough that
bf16 is clean across all three heads.

## Wall-clock (device path vs host-heads path, warm, confidence only)

| N | host-heads | device-resident | delta |
|---|---|---|---|
| 38 (real target) | 22.2 ms | 19.1 ms | +3.1 ms |
| 128 (padded) | 76.3 ms | 41.0 ms | +35.3 ms |
| 256 (padded) | 341.5 ms | 134.6 ms | +206.9 ms (−60%) |

## E2E on the real cached target

`fold(n_step=10, n_sample=1, return_confidence=True)` at `NT=38`: e2e ~5–6 s,
confidence ~23 ms = ~0.5% of e2e. The device port's e2e impact at `NT=38` is
negligible (<0.1%); the win is at large N with `n_sample>1`, where the per-sample
round-trip compounds. The qb1 projection (~4% e2e at `n_sample=25`, `N=256`) is
consistent with the split above but is not re-verified e2e here (no real large-N
target cached; the padded stress is not a real fold).

## Enabling and re-verifying

Set `TT_PROTENIX_CONF_DEVICE=1`. The device path activates only for `NT >= 128`
(the host-heads path runs below that). Reproduce the parity + timing:

```bash
TT_VISIBLE_DEVICES=0 python3 scripts/protenix_confidence_device_parity.py
TT_VISIBLE_DEVICES=0 python3 scripts/protenix_confidence_profile.py
```

Before enabling by default at large N, run a real large-N predict and confirm
pae/pde/plddt PCC vs the host-heads path clears 0.99 (the padded-N numbers above
are a stress test, not a real-target gate).

## What changed

`ConfidenceHead.confidence_device` (and `z_base_device`, `_device_resident`,
`_postprocess`, `device_confidence_enabled`) implement the device-resident path;
`ConfidenceHead.confidence` was refactored to share `_postprocess` (no behavior
change). `Protenix.fold` selects the device path when the flag is set and
`NT >= 128`, otherwise the existing host-heads path. The device path is
feature-detected (skipped if the installed ttnn lacks `clamp`/`ge`/`lt`/`sqrt`/
`embedding`/etc.), following the swiglu-fused-matmul precedent.
