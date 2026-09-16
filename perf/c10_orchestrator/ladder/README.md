# The C10 ladder

Baseline **14.8813 s at 512 aa** and 9.6801 s at 298 aa, pinned 1350 MHz, 16 folds per size
(`c10-bare-baseline` `bc66f7d6d`). Target 10.0 s, so the gap is **4.881 s**.

Every row carries an evidence class, because a number measured on Blackhole at a recorded clock
and one transferred from Wormhole at an unrecorded clock are not the same kind of thing.

| candidate | worth | evidence | status |
|---|---|---|---|
| ttnn trace of the diffusion loop | 3.02 s / 4,077 Mcyc | derived from measured artifacts | queued |
| the rest of the clock-immune term | 0.93 s / 1,256 Mcyc | derived from measured artifacts | **unowned** |
| per-class grid sizing | withdrawn | transferred across architectures | queued |
| `TT_BIO_SDPA_GRID_Q_CHUNK` sign | −0.46 to +0.21 s | not measured anywhere | queued |
| `TT_BIO_HEAD_PAD_TAIL` | 0.21 s / 284 Mcyc | Wormhole, no clock | not rowed |
| `TT_BIO_DIT_FUSED_QKV` | 0.12 s / 162 Mcyc | Wormhole, no clock | not rowed, **excluded** |
| the byte axis, everything remaining | 1.487x ceiling | Blackhole, no clock | closed as a ceiling |

**Priced total 4.16 s, so the fold would read 10.72 s.** The naive sum is 4.28 s, but
`TT_BIO_DIT_FUSED_QKV` and `TT_BIO_HEAD_PAD_TAIL` are jointly 0.713 A against a 0.60 A bar and
cannot both ship, so the cheaper one drops out.

## Read this before quoting the total

**Nothing in this table has been measured as a fold-level win.** Cycles saved to date: zero.
Accuracy spent to date: none. The total is an inventory of what is worth trying, not a forecast.

- Two priced rows are *derived* — arithmetic over measured artifacts rather than measured levers.
- Two are Wormhole numbers that have never run on Blackhole.
- Perturbations stack strongly sub-additively on this fixture, so a stack has to be measured as a
  stack. Summing individual readings is exactly the error the campaign forbids.

**And even at full value the ladder lands at 10.72 s, above the target.** That is the useful
conclusion tonight: on present evidence 10.0 s needs either the clock-immune term to give up more
than the diffusion trace reaches, or a device-work lever nobody has found. The byte axis cannot
supply it — it caps at 1.487x of the floor and existing levers have already consumed an unknown
part of that.

The largest single unowned item is the **0.93 s** of clock-immune cost the trace does not reach:
featurization, MSA handling, output writing and whatever dispatch sits outside the diffusion loop.
`c10-fixed-cost` is measuring the term; nothing is attacking the remainder.

    python3 ladder.py                      # prints ladder.json
    python3 -m pytest test_ladder.py -q    # 9 controls
