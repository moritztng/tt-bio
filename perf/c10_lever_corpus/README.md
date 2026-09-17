# C10 lever corpus

Every perf lever this project measured on Boltz-2, in one machine-readable ledger, each row carrying
the clock it was measured at — or an explicit `unrecorded`.

**The headline: not one of the 27 rows carries a recorded clock.** The whole corpus predates the
2026-09-16 finding that the card's ARC governor sets the fold time on this fixture, so every ratio in
it is a wall-time ratio taken against an unknown denominator between 800 and 1350 MHz.

Nothing here is a speedup claim, no new lever is approved by it, and no production code changed.

## Files

| file | what it is |
|---|---|
| `ledger_src.json` | the source rows. Each one names its quote and the file that quote comes from. Edit this. |
| `corpus.py` | verifies, prices and renders. Refuses any row it cannot ground. |
| `clock_sensitivity.py` | what a wall ratio means at each end of the clock range, per lever class |
| `EXCLUDED.md` | the dead ends, restated so nobody re-runs them |
| `handcheck.py` | re-finds a named sample of rows in their sources by a different route than `corpus.py` uses |
| `out/` | generated: `ledger.json`, `LEDGER.md`, `refusals.json` |
| `tests/` | 25 controls, most of them refusal cases |

## Reproduce

```sh
python3 perf/c10_lever_corpus/corpus.py            # writes out/
python3 perf/c10_lever_corpus/clock_sensitivity.py
python3 perf/c10_lever_corpus/handcheck.py
python3 -m pytest perf/c10_lever_corpus/tests -q
```

State-doc evidence resolves under `~/.coworker/state` by default; override with `--state-root` or
`C10_STATE_ROOT`. Repo evidence resolves against the repo root. CPU only — nothing here opens a
Tenstorrent device, imports a model or touches the network.

## What the extractor refuses

A row reaches the ledger only if every one of these holds. Otherwise it lands in `refusals.json`
with the reason, and it is never silently dropped.

- the evidence file exists and contains the row's quote verbatim, whitespace-normalised
- the row's ratio appears inside that quote, so a row cannot carry a number its own source does not
  say
- the clock is an integer MHz that appears in a clock-evidence quote, or the literal string
  `unrecorded`. There is no inferred clock and no default
- `status` agrees with the flag defaults parsed out of `tt_bio/`, not with prose. A row with no
  `env_flag` must quote the line of `tt_bio/` that sets its default; a row claiming a lever is
  unshipped must name a symbol that is absent from `tt_bio/`, or the un-levered form that is still
  present there
- a `contested` row, one with two readings on record that disagree, is refused rather than
  arbitrated here

## Reading a row

`ratio_scope` is the ruler, and the rulers do not compare. A block ratio is not a step ratio is not
a fold ratio: the 200-step sampler was 5.365 s of the 20.113 s cell those sampler-wall ratios were
taken against, so a sampler-wall ratio is worth roughly a quarter of the same number on the fold.
Do not read down the ratio column and multiply.

`implied_mcycles` is filled in only when the clock is known. Every row here has `clock_band`
instead: what the same ratio would imply at 800 MHz and at 1350 MHz. That is a bracket over an
unknown, not a measurement.

## The clock correction

The campaign opened on the idea that a throttled clock inflates the denominator of a wall-time A/B,
so the byte levers all read small and are worth re-measuring at burst. Under the campaign's own
planning fit `fold_s = 2.901 + 15355/AICLK`, that argument has the wrong sign for the byte class:

- a lever deleting **device cycles** shrinks the `15355/f` term while both arms carry the same
  2.901 s. Throttling inflates numerator and denominator together, so a 3 % cycle deletion reads
  **1.02676x at 800 MHz and 1.02449x at 1350** — 8.5 % *smaller* at burst, not larger.
- a lever deleting **clock-immune seconds** — host work, dispatch, op launch — shrinks the 2.901 s
  instead. That term is 13.1 % of the fold at 800 MHz and 20.3 % at 1350, so the same 0.5 s saving
  reads **1.02315x at 800 MHz and 1.03630x at 1350**, 57 % larger at burst.

So the class an unrecorded throttled clock under-priced is the host/dispatch/launch class. The byte
class was, if anything, marginally over-priced. Converting any unrecorded-clock fold ratio to cycles
carries a flat 9.0 % band across [800, 1350] MHz, which is smaller than most of these levers' own
A/A floors — so the clock ambiguity is not what makes the corpus hard to bank. What makes it hard to
bank is that many of the ratios sit inside or barely above the floor they were measured against.
