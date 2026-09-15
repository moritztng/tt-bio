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

**And main's own tip was folded after the merge.** qb2 rebooted at 02:49Z and left all four cards
free, so `912678a3b` itself — the merge commit, tree verified equal to the branch tree on both hosts
— was folded on card 0 at 512 aa with the flag at its default:

    main tip 912678a3b, boltz2 512 aa, 200 steps / 3 recycles   45781db716ebf020   plddt 0.845919

That is the reference digest, on a good card, from what main actually contains now. It is the one
check the merge could not have before it landed.

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

## 11 of the 44 parity legs did run, on main's own engine code

qb2 rebooted at 02:49Z and freed all four cards. The gate ran on cards 0 and 1 in a bounded 23-minute
window (two cards, not four -- a sibling took card 2 partway through and that is correct). The tree
gated is `d09ba1711`, whose `tt_bio/` is byte-identical to main's tip `912678a3b`, so these verdicts
are main's engine code. Workdir `gate-912678a`, code fingerprint `1bd1584ff974...` over
`['tt_bio', 'scripts']`. Leg JSONs are committed beside this file.

    esmc-300m          dev-vs-ref PCC min 0.99918  mean 0.99947   dev-vs-dev 1.0
    esmc-600m          dev-vs-ref PCC min 0.99943  mean 0.99955
    saprot-35m         R/D emb+logits exactly 1.0, X vs ref 0.99914 / 0.99977
    saprot-650m        R/D emb+logits exactly 1.0, X vs ref 0.99964 / 0.99993
    boltz2-trpcage-nomsa   PASS        boltz2-hsa-nomsa   PASS
    boltz2-prot-nomsa      GAP         boltz2-9ncy-nomsa  GAP
    protenix-prot-msa  kabsch 2.6252 vs floor 2.7631  0.950x  within
    protenix-ubq-msa   kabsch 1.8218 vs floor 1.9234  0.947x  within
    protenix-hsa-msa   kabsch 0.6716 vs floor 0.6951  0.966x  within

Zero regressions. The two boltz-2 GAPs are the pre-existing ones the committed baseline already
records as "reproduces committed", not new. Every Protenix metric reads `within_noise_floor: True`,
and `protenix-ubq-msa` reproduces the earlier pass's 1.822 / 1.923 to four decimals -- the same
numbers off a different card and 27 commits of main later. `protenix-hsa-msa`, which ERRORed on
device contention in an earlier pass, is clean here.

## What is still NOT verified

The remaining 33 legs -- ESMFold2 x3 (in flight when the window closed), OpenFold3, OpenBind,
OpenDDE, RF3 and the rest -- plus `ux_regression.py`, `perf_regression.py` and
`release_gate.py --model size-ladder`, have not been run against this merge. qb1 is hard down and pc's single card is the one
`state/pc-card0-down` excludes from parity work because it miscomputes matmuls nondeterministically,
so a verdict from pc would be worthless in both directions -- worse than no verdict. qb2 is the only
host that can run them, and its cards are contended.

The flag is off on every card and the default path is identical above, so those legs would score
code that cannot behave differently. The legs that WOULD say something new are the flag-ON
cross-model ones, and the honest statement is that only Boltz-2 has been folded with it on.

The open question remains the Wormhole block A/B: 227.5 GB/s against Blackhole's 424.7 makes the
deleted bytes worth ~1.9x more there and the sign may differ. Needs whglx. Ceiling measured at
1.03200x (`block_ablate_512_whglx_c2.json`); the A/B with the kernel built has never been run.
