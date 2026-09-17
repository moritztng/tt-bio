# The release gate never folds Boltz-2 at production length above 117 residues

`c10-size-scaling` wedged a Blackhole p300c on a **768 aa Boltz-2 fold at shipped settings** — 200
sampling steps, 3 recycles, a 35-row A3M — nondeterministically, after five identical folds had
completed. 768 aa is a size a user can ask for. So: would the release gate have caught it?

No. And not because any arm is broken.

## The gate's Boltz-2 coverage, on the two axes that matter here

| leg | target | size | steps |
|---|---|---|---|
| accuracy (per-model fold) | `examples/prot.yaml` | **117 aa** | **200** |
| l1-budget | `examples/affinity_fkg.yaml` | 107 aa | 200 |
| size-ladder | `cdk2x2_{256,512,640,768}` | **up to 768 aa** | **6**, single-sequence |
| capacity | abag yamls | 1095 / 891 aa | 6 — and protenix-v2 / opendde-abag, not Boltz-2 |

**Size coverage and length coverage come from different legs and they do not overlap.** Every
Boltz-2 target above 117 aa is folded at 6 steps — **33.3× shorter** than the 200 the product runs —
and single-sequence, so the MSA path is not exercised at those sizes at all.

The wedge happened at 768 aa and 200 steps. The gate's only Boltz-2 coverage at that size runs
roughly **3 % of the diffusion device work**. A hang that needs production-length work to appear is
invisible to it, and this one is nondeterministic on top.

## This is not a criticism of any arm

The size-ladder arm's own docstring says it counts guard decisions rather than trajectory
statistics, picks six steps deliberately, and names its own price — "a cliff living ONLY in the MSA
module is invisible here". It is doing exactly what it claims. The hole is in the **composition**: no
leg folds a large target for a long time, because no leg was ever asked to.

## The minimal fix, which is not implemented here

- **One long rung**: fold the top size-ladder rung at production steps with an MSA, scored on
  completion with a timeout rather than on counters. Cost is one fold.
- **Or say so**: state the limit in the gate's own summary, so "size-ladder PASS" is not read as
  "large targets are safe".

Either is a release-gate change and therefore gated. It stays on this branch and it is Moritz's
call, not this row's.

## Limits

Coverage is parsed from `scripts/release_gate.py`'s constants and from the fixture YAMLs, so the map
tracks the gate rather than a memory of it — a control fails if any constant stops parsing, and
another fails if the gap is ever closed. A leg folding a large target through a path this parse does
not recognise would be missed; the leg list is explicit so it can be checked by eye.

This says nothing about whether the wedge reproduces or what causes it. **One occurrence in six
folds is not a rate.** And rf3 does fold 997 aa in the gate, but that is a different model.

    python3 gate_coverage.py                      # writes gate_coverage.json
    python3 -m pytest test_gate_coverage.py -q    # 11 controls
