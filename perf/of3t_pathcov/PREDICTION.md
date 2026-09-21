# of3t-pathcov — registered before the first census run

Written 2026-09-21, before any run of `census.py`. The static inventory in `sitecov.py` had
been run (it parses source and opens no device); nothing dynamic had.

## What the instrument measures

Three quantities in one taped forward+backward, per `ttnn.<verb>(...)` call site in `tt_bio`:

- **executed** — the site's line fired while a tape was open (`sys.monitoring` LINE events,
  armed on `_swap(True)` and cleared on `_swap(False)`).
- **taped** — the proxy's verb wrapper was entered from that site, counted by caller frame.
- **reach** — the parameter leaves upstream of the site's output, carried on each tape node as
  a bitmask built from its parents.

`executed AND NOT taped` is the A2 hole as a measurement: the site ran inside a tape and never
reached the proxy, which is what a module-global rebinding cannot do for a name that is not a
module global.

## A31 check, before the numbers

The obvious phrasing of deliverable 1 — "the share of the squared gradient norm carried by the
never-executed sites" — **is not satisfiable by any run**. A site that does not execute produces
no node, no gradient and no mass, so its share is identically 0 for arithmetic reasons, and the
number would say nothing. The measurable form, and the one scored below, is the complement:

> the share of the reference squared gradient norm held by parameters that **no executed site
> reaches**.

That is a property of a run that happened, it composes with the 92.1568 % because it is scored
in the same float64 denominator (`MODEL_shipped.json`'s `model_squared_gradient_norm_measured`
= 10.279642678524981 over 4170 reference tensors), and a never-executed branch shows up in it
exactly when it owns mass nothing else produces.

## The numbers I expect

Static inventory, already taken: **3227** `ttnn.*` call sites in `tt_bio` (excluding `_vendor`),
**2531** of them on a verb that has a tape entry. `tenstorrent.py` holds 945 (840 taped-verb),
the nine `openfold3_*` modules 361 (326), and the rest are other models.

Arm: `perf/of3t_diffusion/device_gradient.py --structs all`, the shipped arm behind
`MODEL_shipped.json`'s diffusion scope, 48 structures, card 2 qb2 p300c.

| # | prediction | value |
|---|---|---|
| P1 | taped-verb sites in the SHIMMED module set (the census universe) | 1400 ± 400 |
| P2 | of those, executed inside a tape | 25 % ± 10 pp |
| P3 | so: never executed | 75 % of the universe |
| P4 | sites that executed inside a tape and were never taped (the A2 hole) | **0** |
| P5 | the same count at `5a2efa001^`, before A2's two sites were fixed | **≥ 1**, and the guard names both files |
| P6 | parameters reached by ≥1 executed site, as a share of the arm's own leaves | > 95 % |
| P7 | reference mass held by parameters no executed site reaches, model denominator | 7.8 % ± 1 pp, i.e. the complement of 92.1568 to within the arm's own scope |
| P8 | the worst uncovered branch by gradient mass | `pairformer_stack` at 5.8282 %, uncovered in the model union because its arm is driven by a different boundary, not because it never runs |
| P9 | the largest branch that genuinely never runs in any arm | `input_embedder` at 0.8007 % |

P4 and P5 are the pair that makes the guard worth landing: an assert that has never failed has
tested nothing, so the same guard is run against the pre-fix tree and must fire there.

P8 and P9 are deliberately separated. "Uncovered" in the equivalence sense (not in the 92.1568 %)
and "never executed" in the path sense are different claims, and the campaign's own habit is to
quote one as the other.

## What would falsify the instrument rather than the model

- A non-zero `untracked_taped_calls` count means the frame's line does not land inside the static
  site's span, so the attribution is wrong rather than the coverage. Reported, not hidden.
- `n_tape_opens == 0` means the arm never opened a tape and the whole census is vacuous.
- `leaf_names` all `None` means the instrument's name map was not found, and the mass cannot be
  scored in the reference denominator. The site census still stands; the mass share does not.
