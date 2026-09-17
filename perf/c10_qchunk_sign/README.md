# Does `TT_BIO_SDPA_GRID_Q_CHUNK` earn its default?

It is ON for every user today, and the project had two readings of its value that disagreed in
sign. `c10-lever-corpus` therefore refused to price it and carried it as contested, somewhere
between -623 and +281 Mcycles. This row measures the sign.

**It is positive at both sizes, and it is worth far more at 298 aa than anyone had priced.**
On a Blackhole p300c at a pinned 1350 MHz, Boltz-2, 200 sampling steps, 3 recycles, seed 0:

| size | flag ON | flag OFF | ON is faster by | ratio | Mcycles at 1350 | A/A floor |
|---|---:|---:|---:|---:|---:|---:|
| 512 aa | 14.888 s | 14.974 s | **0.064 s** [0.056, 0.127] | 1.00431x | +87 | 0.055 s |
| 298 aa | 9.667 s | 10.492 s | **0.829 s** [0.811, 0.846] | 1.08568x | +1119 | 0.043 s |

Both arms wrote a byte-identical CIF, so there is no accuracy question to answer. **No production
file changed and no default was touched.** Landing anything is a separate decision; this row only
fixes the record.

## Why 298 aa is 13x the 512 aa win

The flag picks the widest `q_chunk` whose work units still fit one pass of the compute grid,
instead of the fixed 256 cap. The per-call chunk census, taken on a warm-up fold in each arm,
shows what it actually does:

| size | site | calls/fold | 256-cap picks | rule picks | work units on 110 cores |
|---|---|---:|---:|---:|---|
| 512 aa | `token_dit` | 4800 | 256 | 128 | 32 -> 64 |
| 512 aa | `atom` | 1200 | 32 | 32 (declines) | 560, unchanged |
| 298 aa | `token_dit` | 4800 | 256 | **64** | 32 -> 80 |
| 298 aa | `atom` | 1200 | 32 | 32 (declines) | 336, unchanged |

One site moves and one declines, at both sizes. The atom site has 32 query rows, one tile, so it
has nothing to split and both arms get a byte-identical program config; that makes it a free
correctness control rather than an assumption.

The size gap is a divisibility effect, not just an occupancy one. At 512 aa the 256 cap divides
the padded q axis, so the cap costs only idle cores. At 298 aa the tokens pad to 320, which 256
does **not** divide, so the capped arm runs 2 chunks of 256 over a 320-row axis: 16 work units on
110 cores, and a padded tail on top. The rule picks 64, which divides 320 exactly and fills 80
units. That is why the same flag is worth 0.43 % at one size and 8.6 % at the other.

The general form, because it will recur: **pricing a grid-tuned chunk flag only at a size where
its cap happens to divide the padded axis underprices it by an order of magnitude.**

## What this settles, and what it does not

The two readings on record are now both explained rather than arbitrated. The published
`+0.496 pp` came from a different branch and a truncated precursor fold. The `-0.122 +/- 0.103 pp`
that `k10-transfer-function` measured over 44 Blackhole folds is a 512 aa number, and 0.12 pp at
512 aa sits inside the +/- 0.57 pp A/A band this session itself shows at that size. Neither reading
had the resolution to sign a 0.43 pp effect. `b2z2_qchunk`'s 1.00355x at an unrecorded clock and
this row's 1.00431x at 1350 MHz agree.

The 512 aa **sign** is solid: 8 of 8 drift-immune rep contrasts positive, 14 of 16 adjacent pairs
positive, both confidence intervals excluding zero. The 512 aa **magnitude** is soft, because
0.064 s is about the same size as this session's own single-pair A/A spread of 0.083 s; read it as
0.05 to 0.13 s rather than as a point. The 298 aa result needs no such hedge: it is 48x the
session's A/A spread and 16 of 16 pairs agree.

Only one session ran, on one card. Nothing here measures device cycles, per-op cost, or any other
size.

## How it was measured

Both arms are interleaved inside **one process and one device context**, `on off off on` per rep,
8 reps, 16 timed folds per arm per size. The lever sits at positions 1 and 4 of every rep, so any
linear drift across a rep cancels exactly in the rep contrast. The A/A control is the middle
`off|off` pair of each rep plus the `on|on` rep boundary, 15 pairs drawn from the same rep
sequence rather than from a second session. Each arm gets one census fold before any timing, so
neither pays first-mover program-config build inside the measurement.

The timer spans the complete `predict_one` including featurization and CIF writing, with a device
synchronize on both sides. The clock was forced to 1350 MHz and sampled **during** every fold by a
separate process with monotonic read brackets: minimum = maximum = 1350 MHz across all 68 folds,
449,883 during-samples at 512 aa and 303,566 at 298 aa, zero read errors. Device holders were
sampled at 10 Hz over all four chips and host CPU at 4 Hz over the whole locked window; no foreign
device holder at any point, loudest foreign host process 11.7 % of a core, peak load 1.51. The
above-cap fused-SDPA route counters read `[0, 0]` in every fold, so nothing here can be attributed
to that route.

Parity is byte-level: one CIF sha256 across all 32 accepted folds of both arms at each size, 256
cross-arm float64 Kabsch comparisons per size with a maximum of 3.7e-15 A (512 aa, 0.60 A bar) and
7.1e-15 A (298 aa, 0.35 A bar), identical pLDDT. `q_chunk` partitions independent query rows, so
this is what the code predicts.

`controls.py` holds 18 known-answer checks on the parts this harness adds: the schedule, the
pairing, the sign convention in both directions, the drift immunity of the rep contrast against a
blocked order that fakes a 0.160 s lever out of pure drift, the census gate, and the shipped chunk
rule replayed on CPU. The timer, clock, holder, geometry and host-CPU controls come from
`c10-bare-baseline`, whose modules are imported unmodified and pinned by sha256 in `imports.json`.

## Reproduce

`run.sh` sets its own environment. The three steps after it need the tt-bio venv and this
directory on `PYTHONPATH`, otherwise they stop at `ModuleNotFoundError: numpy`:

    bash perf/c10_qchunk_sign/run.sh <name>

    export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
    export PYTHONPATH=$PWD/perf/c10_qchunk_sign
    python3 perf/c10_qchunk_sign/reduce.py perf/c10_qchunk_sign/runs/<name> \
        --out perf/c10_qchunk_sign/runs/<name>/analysis.json
    python3 perf/c10_qchunk_sign/summary.py perf/c10_qchunk_sign/runs/<name>/analysis.json
    python3 perf/c10_qchunk_sign/controls.py | tee perf/c10_qchunk_sign/controls.log

Only the first step needs a card. The other three are CPU-only and rerun against the committed
`runs/qsign1/`: the reducer rebuilds `analysis.json` byte-identically and `controls.py` reprints
the 18 lines in `controls.log`.

The reducer exits non-zero unless every criterion in `criterion.json` holds. `prediction.json` was
written before the first fold; it got the sign right at both sizes, the 512 aa magnitude right
(predicted 0.045 s, measured 0.064 s) and the 298 aa magnitude wrong by 28x (predicted 0.030 s,
measured 0.829 s), because it priced occupancy and missed the divisibility term.
