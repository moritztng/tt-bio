# What the re-merge is verified against, and what it is not

The gate epilogue forked 24 commits behind main and the merge conflicted in `tt_bio/tenstorrent.py`
and `tt_bio/triatt_sdpa.py`. This file records what the hand resolution was checked against, so the
next reader does not have to reconstruct it from commit messages.

Branch tip at the time of writing: the merge of `origin/main` at `7ab9dc7bb`.

## The default path is unchanged, by compilation and by measurement

**Compilation.** Every line the branch adds to `tt_bio/kernels/triatt_sdpa/` sits inside
`#ifdef GATE_EPILOGUE`, and the branch deletes none. The host only defines `GATE_EPILOGUE` when a
gate tensor reaches `sdpa_generic.build`, which needs the flag on. So with the flag off the kernel
sources preprocess to exactly main's and the compiled binary is identical -- for all five models
that route through this kernel, not just the one that was folded.

**Python.** The diff to the three Python files is additive: a `gate=None` keyword threaded through
`_tri_att_sdpa_at`, `triatt_sdpa.sdpa` and `sdpa_generic.build`/`sdpa`, plus `if not gate_fused:`
around the multiply in `gate_and_project`. With the flag off `_tri_att_gated_sdpa` returns `None`
at its first guard without touching a counter, `gate_fused` stays `False`, and the call that runs
is the same `_tri_att_sdpa(q, k, v, b, scale, ckc, ragged_pad)` main makes.

**Measurement.** `git archive origin/main` extracted outside the repo and folded on the same card in
the same session window as the merged tree (`fold_mainref_qb2_c2.json`):

    298 aa   main a8c6fd65f70f4418   merged flag OFF a8c6fd65f70f4418   merged flag ON a8c6fd65f70f4418
    512 aa   main 45781db716ebf020   merged flag OFF 45781db716ebf020   merged flag ON 45781db716ebf020

All six committed files under `cif/` still hash to those two digests, so the record reproduces off
the checkout with no card at all.

## The card-free suite matches main leg for leg

`pytest -q` with no device: **7 failed, 3574 passed, 183 skipped**. The same suite on pristine
`origin/main` fails 6 of those 7 identically (`test_capacity_gate` x3, `test_perf_citations` x2,
`test_size_ladder_gate`). The 7th, `test_repo_root_clean`, needs care: it PASSES on a `git archive`
control, because `_tracked_root` asks git and an extracted tarball has no `.git` to ask. It is
vacuous there, not green. `git ls-tree -d origin/main` lists both `artifacts` and `patches` at the
root, so that failure is main's too.

**A `git archive` extract is not a control for any test that queries git.** It answers with silence
and the test reads silence as a pass.

`scripts/packaging_smoke.py`: GATE PASS, 52 data files, 44 deps, 49 imports. The epilogue adds no
new kernel file -- it edits two that already ship -- so the packaging glob is not in play here.

## What is NOT verified

The 44-leg `scripts/full_parity_gate.py`, `ux_regression.py`, `perf_regression.py` and
`release_gate.py --model size-ladder` have not been run against this merge. No card in the fleet can
currently host them: qb1 is hard down, qb2's four cards are held by three sibling workers' own
gates, and pc's single card is the one `state/pc-card0-down` excludes from parity work because it
miscomputes matmuls nondeterministically. A verdict from pc would be worthless in both directions,
which is worse than no verdict.

The flag is off on every card and the default path is identical above, so those legs would score
code that cannot behave differently. The legs that WOULD say something new are the flag-ON
cross-model ones, and the honest statement is that only Boltz-2 has been folded with it on.

The open question remains the Wormhole block A/B: 227.5 GB/s against Blackhole's 424.7 makes the
deleted bytes worth ~1.9x more there and the sign may differ. Needs whglx. Ceiling measured at
1.03200x (`block_ablate_512_whglx_c2.json`); the A/B with the kernel built has never been run.
