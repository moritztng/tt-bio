We put proga's tt-bio backend on a card. Short version: it works against both oracles, every key your readers consume is there, we found one real bug on your side that would have quietly ruined a population run, and the link is written.

## What we ran

qb1 Blackhole p150a, card 3, AICLK 1350 MHz sampled every 5 s during the folds (50 samples at 1350, 6 at 1343). tt-bio tree `1e3b37664`, proga pinned at `8a6c8ab`. Target human ubiquitin (76 aa) plus a 24-mer de-novo binder, 100 tokens, two candidates per oracle, with exactly the command line `TTBioOracle.predict_complex` builds: `--write_pae --output_format cif --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --seed 0`, and the same content-addressed a3m your `_write_yaml` emits.

| oracle | wall per candidate | status |
|---|---|---|
| `boltz2` | 37 s, then 35 s | ok |
| `protenix-v2` | 150 s, then 59 s | ok |

To be precise about what executed: we drove `tt-bio predict` with proga's exact argv and yaml, and checked the outputs against your readers with our own copy of their logic, rather than running proga's driver on a shared multi-tenant fleet. So these numbers are the oracle's per-candidate cost, not a GA end to end.

Wall includes model load, which dominates at 100 tokens: the boltz2 fold itself is 11-13 s, the rest is loading the model once per `tt-bio predict` call. protenix-v2's first candidate also pays a cold MSA search (44.7 s of the 150 s).

Every reader in `ttbio_oracle.py` matches the real output, on both oracles:

- `*_pae.npz` carries `pae`, 100x100 after your `ndim == 3` squeeze. boltz2 also writes `pde`, `contact_probs` and `contact_cutoff_A` in the same file; protenix-v2 writes `pae` and `pde`.
- `results.json` row has `id`, `status: "ok"`, `iptm`, `pair_chains_iptm` and `chains_ptm`, and **the chain keys are integer strings on both models**, so your `int(i)` is safe. That was the thing most worth confirming, since it is the path we pointed you at.
- pLDDT is 0-100 in the CA B-factor, so your `/100.0` is right, and the cif chain names match the yaml ids you index by.
- `_find_pred_dir` does the work: tt-bio writes `structures/` under `<out_dir>/<model>_results_<name>/`, so your first guess at `run_dir/structures` misses and the cif-parent fallback catches it. No change needed, just so you know which branch is live.

## The bug: every candidate after the first is scored with the first one's structure

`cli/design.py:380` calls `predict_complex` with no `dump_dir` and no `name`, so the whole population folds into one run dir under the name `design`. tt-bio skips an input whose output already exists in `--out_dir`. We folded two different binders into the same out_dir and the second produced no work at all, just `All predictions complete`, with `results.json` still holding the first binder's `iptm` of 0.562011. The second binder's real iptm, from a clean dir, is 0.763021.

Nothing errors. The GA runs, generations tick by, and every evaluation after the first returns generation 1's score.

Your `BoltzOracle` already handles this: it passes `--override` to `boltz predict`. `TTBioOracle`'s command dropped it. tt-bio has the same flag with the same meaning, so adding `"--override"` to the list at `ttbio_oracle.py:49-60` fixes it. Giving each candidate its own `dump_dir` or `name` would work too, and would cost you a little more disk but give you a browsable trajectory.

We would rather you fix it in proga than us change tt-bio: skipping existing outputs is deliberate, and it is what makes a 500-target run resumable after a crash.

## Two smaller things

**MSA across a population.** On boltz2 your cache works exactly as intended: the target a3m is written once and every later candidate reports `MSA ready in 0.0 s`. protenix-v2 does not take that shortcut, it runs its own paired MSA search per candidate (44.7 s cold, 9.4 s warm here) because it needs paired MSAs for a multi-chain input. If you drive a population through protenix-v2, pass `--msa_dir` and `--msa_cache_only` through `ttbio_extra_args`.

**A contact term is available after all.** Your docstring says tt-bio emits no distogram, so contact terms are skipped. True for logits, but with `--write_pae` boltz2 also writes `contact_probs` and `contact_cutoff_A` into the same npz. You cannot rebuild logits from that, but a contact term can read the probabilities directly if you want one.

## The link

tt-bio's README gets a **Community projects** section and proga is the first entry. It credits you, says what the tool does, and says we drove tt-bio with proga's own oracle command line against boltz2 and protenix-v2 on a Blackhole card. It is a good piece of work and the fit with tt-bio is genuinely clean.

It is written and pushed on `wk/proga-review` and goes to `main` with the next merge; we will come back and close this issue with the link once it is there, so you get a notification rather than having to check. If you want to send a docs PR adding detail, we will take it.

The keys we document stay stable: `confidence_score`, `ptm`, `iptm`, `plddt`, `chains_ptm`, `pair_chains_iptm`, the pae npz and its json sidecar. If one of those breaks or changes shape under you, that is a bug on our side and we will fix it.

One last thought: for population-scale runs where you are throughput-bound rather than latency-bound, proga can point at JapanFold's hosted API instead of a local card (docs.japanfold.aiand.com). Same models, no card to manage, and a GA that wants many independent folds suits it well.

Thanks for building this, and sorry again for the two-minute close.
