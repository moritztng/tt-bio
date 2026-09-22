# D117: multi-chain permutation alignment

The OF3T campaign recorded upstream's multi-chain permutation alignment as NOT COVERED, on the
grounds that `safe_multi_chain_permutation_alignment` raises `KeyError 'ref_space_uid_to_perm'`
and takes its documented fallback to naive alignment. It does not. On both batches this campaign
holds it **completes**, inside the real checkpointed forward, and it resolves atom symmetry the
fallback leaves alone. The two ways the campaign drove it into the fallback are both harness
defects, and both are reproduced here as controls.

Upstream's `permutation_alignment.py` and `model.py` are not modified. Nothing here changes a
shipped default.

    bash permalign_run.sh        # ten arms, the check, the table

| file | what |
|---|---|
| `permalign_probe.py` | the probe. Builds a batch, calls upstream's own wrapper, reports which path ran |
| `permalign_guard.py` | the detector: a handler on upstream's logger plus pass-through wrappers, cross-checked |
| `permalign_selftest.py` | the proposed check — log scan, artifact assertion, and a canary that drives upstream into both catch tiers |
| `permalign_run.sh` | every arm, one line each |
| `summarise.py` | renders the arm table from `runs/` |
| `runs/` | one JSON per arm, plus the stderr-visibility evidence |
| `census_correction.json` | the edit `perf/of3t_gradients/coverage_census.py` needs, and why this row did not make it |

The three arms CONFIGURED to break the alignment must fall back and the rest must complete;
`permalign_selftest.py` asserts exactly that, reading each arm's own recorded configuration
rather than its tag, because a probe that cannot report both outcomes has measured nothing.

Needs `/home/moritz/.coworker/scratch/of3t-bondcov/{of3pkg043,datasets,batch_step003.pt}` and
`/home/moritz/.boltz/of3-p2-155k.pt`. The seven arms without `--via-forward` need no checkpoint.
