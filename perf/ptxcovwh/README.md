# ptxcovwh — Protenix-v2 at the wh-galaxy bar

Re-measures `PROVEN["protenix-v2"]` on the serving Galaxy. The entry was stamped at `4fbc152f`
and 229 commits had touched `SIZE_AFFECTING_PATHS` since, so it described a tree we no longer
ship.

Result: 1024 tokens at 8832 alignment rows folds in 655.1 s on `72596df4f`, against 705.0 s on
the live `engine.pin` and 730.5 s recorded on 2026-09-08. Zero backbone breaks, all residues
delivered, and the same single 2147483648 B DRAM refusal absorbed on both engines.

The leg worth keeping is `p1024`. Running the pinned engine on the original fixture reproduces
the 2026-09-08 structure byte for byte, which is what makes main's 0.171 A deviation readable as
main's rather than the harness's. Always run it alongside `m1024`, and always run `m1024s7` too:
an unpaired cross-engine RMSD carries the full seed floor, and here that floor is 1.471 A.

Fixture `ptx_1024.yaml` is the 2026-09-08 one — a 1024 aa CDK2 tandem tiling, apo, so tokens
equal residues. Its alignment is not in this repo; `leg.sh` points at the cached a3m on GWH02
and passes `--msa_cache_only` so a missing cache fails loudly instead of folding single-sequence.

    ./leg.sh m1024   ptx_1024 0 16 main 0
    ./leg.sh p1024   ptx_1024 1 17 prod 0
    ./leg.sh m1024s7 ptx_1024 3 19 main 7
    ./collect.sh

Cards are UMD indices; the second number is the `/dev/tenstorrent` node, which is a different
permutation on every host. Resolve it with `tt_bio.runtime.umd_index_to_dev_node()` rather than
assuming they match, and only ever use cards outside the serving pool.

Measurements in `results.jsonl`, write-up in `state/cov-stale-protenixv2-whgalaxy.md`.
