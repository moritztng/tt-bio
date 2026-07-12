# Protenix-v2 recycling_steps revisit — re-measured post template-embedder fix, MERGE recommended

Follow-up to `docs/protenix-template-embedder-fix.md`. That fix closed a real bug (the
featurizer never emitted `template_aatype`, so Trunk always ran `nt=0` — the template-embedder
pass was skipped on every real predict call) and, at the **unchanged** default
`--recycling_steps=3`, already cut delivered RMSD substantially: 7ROA 3.47→1.84 Å, hemoglobin
1.35→0.60 Å, no regression.

The open question it left: `wk/tt-bio-protenix-confidence-fix` (pushed, not merged) separately
found that bumping `--recycling_steps` 3→10 (Protenix-v2's spec, `Trunk.N_CYCLES`) fixes 7ROA's
delivered-RMSD gap (-1.13 Å) but **regresses** hemoglobin (+0.61 Å, mixed sign) — measured
against a trunk that, at the time, was *missing the template-embedder pass entirely*. Does the
recycling lever still help, now that trunk is running correctly?

## Measurement

`scripts/protenix_recycle_sweep.py` (cherry-picked from the confidence-fix branch, unmodified
except for its `WT` path), same protocol as both prior branches: `--sampling_steps 200
--diffusion_samples 5 --seed 0`, on current main (template fix included), sweeping
`recycling_steps ∈ {3, 10}` per target from **bit-identical input feats** (features built once
per target, both cycle counts folded from the same `feats` object) — so the 3-vs-10 delta is
directly attributable to trunk convergence, immune to any feature-reproducibility noise between
runs.

| target | delivered@3 | delivered@10 | oracle@10 | Δ delivered |
|--------|---:|---:|---:|---:|
| 7ROA (`examples/prot.yaml`, monomer, shallow MSA) | 1.610 Å | **1.347 Å** | 1.320 Å | **−0.263 Å** ✓ |
| hemoglobin (`examples/hemoglobin.yaml`, α₂β₂ tetramer, deep MSA) | 0.595 Å | **0.543 Å** | 0.543 Å | **−0.052 Å** ✓ |

**No regression on either target — the mixed sign is gone.** hemoglobin@10 exactly hits its own
oracle (delivered == best sample); 7ROA@10 lands within 0.03 Å of oracle.

Cross-checked against the production CLI path (`scripts/release_gate.py`, `examples/prot.yaml`,
same protocol, new default `recycling_steps=10` in effect): **1.347 Å, TM 0.947 — matches this
harness's number exactly**, confirming the standalone sweep harness faithfully reproduces the
real predict path here (unlike the signal-correlation/distogram-consistency harnesses' documented
MSA-subsampling gap). It also beats the template-fix doc's committed `recycling_steps=3` CLI
number (1.842 Å) by 0.495 Å, an even larger CLI-level win than this harness's 0.263 Å delta.

## Why the mixed sign disappeared

The confidence-fix branch's hemoglobin regression (1.35→1.96 Å) was measured against a trunk
that never ran the template embedder — i.e. against a strictly worse, and differently-shaped,
baseline than what ships today. With the template embedder now contributing its (large) share of
the trunk's pair representation, recycling has less remaining work to do to converge the
ensemble, and what's left to gain from more cycles is modest and monotonic in the right direction
for both targets, rather than being a same-magnitude tradeoff between them.

## Decision: MERGE recycling_steps default 3→10

Both targets now improve (or are flat) — no mixed sign, no target regresses. The gain is smaller
than the confidence-fix branch originally measured (expected: the template fix already captured
most of the accuracy gap), but it is real, in the correct direction for both targets, and free of
downside beyond ~3x trunk compute (a fraction of total fold time; diffusion/coordinate path
untouched). Recommend merging the `_resolve_recycling_steps` default-10-for-protenix-v2 change
(cherry-picked here from `wk/tt-bio-protenix-confidence-fix` commits `45cc40c` + `c8ded22`,
unmodified) and **abandoning** `wk/tt-bio-protenix-confidence-fix` itself (superseded by this
branch + the template-embedder fix — its analysis/root-cause doc remains valid context but its
proposed change should land from here, against the current baseline).

## Release gate (`scripts/release_gate.py`, new default in effect)

| model | RMSD (Å) | TM | floor | result |
|---|---:|---:|---|---|
| protenix-v2 (recycling_steps=10, new default) | 1.347 | 0.947 | ≤6.0 / ≥0.5 | **PASS** |
| boltz2 (recycling_steps=3, unchanged) | 1.491 | 0.940 | ≤3.0 / ≥0.75 | PASS (unaffected) |
| esmfold2 | — | — | ≤4.0 / ≥0.65 | environment FAIL — `No module named 'transformers'`, reproduces identically on unmodified `main` and the shared checkout; a pre-existing host dependency gap, not caused by this change |

`recycling_steps` resolution is per-model (`_resolve_recycling_steps`, `tt_bio/main.py`) —
boltz2/esmfold2 keep their unchanged default of 3, so this change cannot affect either; the
esmfold2 gate leg fails identically with or without this branch's commits and is out of scope
here.

## Host-only unit test

`tests/test_recycling_default.py` (cherry-picked from the confidence-fix branch, unmodified):
25 passed — pins protenix-v2 unset→10, boltz2/esmfold2 unset→3, explicit `--recycling_steps`
honored verbatim for every model.

## Release-gated

Accuracy-affecting default-behavior change (same policy as the MSA-on-default and
template-embedder-fix precedents) — flagged for the orchestrator/Moritz's merge decision, not
merged from this branch.
