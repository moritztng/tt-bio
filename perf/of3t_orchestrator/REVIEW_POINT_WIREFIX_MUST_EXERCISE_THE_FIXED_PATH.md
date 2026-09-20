# Review point to settle before `of3t-wirefix` is accepted (pass 182, row still live)

`of3t-wirefix` fixed the four §7 wiring divergences in `tt_bio/train/recipes.py` and
`tt_bio/train/optim.py` (commit `7a6d1b029`), with upstream citations and a wiring test whose
five assertions all fail when reverted. Two checks I ran as orchestrator, and one question the
row has to answer.

## Checked, clean

**No cross-model blast radius.** The changed defaults (`weight_decay=0.0`,
`plateau_until=50000`) are on `train_loop`, and `train_loop` has **no production callers** —
`git grep` finds it only in tests, docs and this campaign's own harness. So the default flip
cannot silently move Protenix's or Boltz's schedule, which was the thing to rule out. The
design is also right on UNIFIED-NOT-PER-MODEL: both are *arguments* with a documented
`plateau_until=None` selecting the Protenix family, not a per-model branch.

**`AdamW`'s own default is untouched at `weight_decay=0.01`** (`optim.py:132`), which is
correct — the torch convention belongs to the class, and it is the *recipe* that has to match
upstream's plain `torch.optim.Adam`. But it does mean the divergence is closed **only on the
path that passes 0.0 explicitly**, so anything constructing `AdamW` directly still gets 0.01.

## The open question

`perf/of3t_traj20/traj20.py` does construct `AdamW` directly. Line 210 reads *"`recipes.py:118`
verbatim: no weight_decay, no clip_norm, no betas, no plateau_until"* and line 219 sets
`kw["weight_decay"] = 0.0` only for the arms that want it. That is a legitimate A/B design —
the arms have to differ somewhere — but it means **the harness reproduces what the fix should
do rather than executing the fixed `train_loop`.** A re-run built that way measures the
replica, not the shipped path, and the two are free to drift the moment either is edited.

This is the campaign's own recorded trap twice over: `parity-gate-builds-both-sides-from-the-
same-source`, and `verify-the-deployed-artifact-not-your-own-change`.

**What would settle it.** Either (a) the `wired` arm calls the real `train_loop`, or (b) the
row shows the replica and `train_loop` cannot drift — e.g. the wiring test pins the
construction in `recipes.py` *and* the harness derives its kwargs from that same source rather
than restating them. The existing `test_of3_wiring_matches_openfold3.py` asserts call sites and
defaults in `train_loop`'s source, which is most of (b) but not the harness half of it.

**Not raised with the row mid-run.** It is measuring; it pre-registered its arms and bars
before touching the source, which is the right order, and it may already be handling this in
the part it has not pushed. This is to be checked against its conclusion, not shouted into a
live measurement.

---

## RESOLVED by the row itself, in `fe693257a`, before I raised it

`of3t-wirefix` closed this independently and more precisely than I had framed it. `traj20.py`
now carries:

```python
def _shipped_defaults():
    import inspect
    from tt_bio.train.recipes import train_loop
    return {k: v.default for k, v in inspect.signature(train_loop).parameters.items()}
```

and the `fixed` arm takes `weight_decay` and `plateau_until` from it rather than restating
them, with `shipped_defaults` recorded into the result file so the arm carries the identity of
the source it claims to measure. The row's own comment gives the reason in the same terms:
*"the arm and the thing it claims to measure can drift apart without either changing."*

**Coverage is now 3 of 4 exercised or source-derived, and the fourth is pinned — worth stating
exactly rather than calling it closed outright:**

| fix | how the harness now relates to the shipped source |
|---|---|
| 1. `weight_decay=0.0` | **read** from `train_loop`'s signature |
| 4. `plateau_until=50000` | **read** from `train_loop`'s signature |
| 3. participation-count division | **executed** — it lives inside `AdamW.step()`, which the harness calls |
| 2. per-sample clipping loop | still **replicated** — it is control flow in `train_loop`'s *body*, not a default, so a signature read cannot reach it; pinned instead by `test_of3_wiring_matches_openfold3.py` asserting the call site in that body |

Fix 2 is the only one where the harness and the shipped path remain two pieces of code that
have to agree, and it is the one with a source-asserting test beside it. That is a defensible
position and the row should say so in its conclusion rather than implying all four are
executed. No action for the row beyond that.
