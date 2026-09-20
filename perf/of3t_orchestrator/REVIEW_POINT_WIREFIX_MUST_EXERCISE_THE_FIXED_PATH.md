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
