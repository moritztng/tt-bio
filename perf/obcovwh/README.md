# obcovwh — OpenBind-0 capacity on the Wormhole Galaxy

Re-measurement of `PROVEN["openbind"]` for `wh-galaxy` after 260 commits of allocation-shape
work had made the 2026-09-07 reading describe a tree we no longer ship
(`ws:cov-stale-openbind-whgalaxy`, state doc in `~/.coworker/state/`).

`leg.sh` folds one rung on one pinned card. The engine is selected by `PYTHONPATH` and
**asserted** before the card is spent: a copied venv silently repoints production, and an arm
that imports prod agrees with itself.

    ./leg.sh m1024apo ob_apo_tile_1024 0 16 main      # 1024 tokens, origin/main
    ./leg.sh p1024apo ob_apo_tile_1024 1 17 prod      # 1024 tokens, live engine.pin
    ./leg.sh m1088lig ob_lig_tile_1024 3 19 main      # 1088 tokens, expected OOM

The card index and the `/dev/tenstorrent` node are **both** arguments because on this Galaxy the
two numberings are a permutation with no fixed point: `TT_VISIBLE_DEVICES` and the device lease
name a UMD index, while `lsof` and the fd-level collision live on a node. Resolve the map with
`tt_bio.runtime.umd_index_to_dev_node()` rather than assuming it.

`TT_BIO_SIZE_LIMIT=0` is set because the shipped guard caps openbind at 960 residues, which is
exactly what the ceiling under test is above.

Results in `results.jsonl`, one object per leg. Both axes are recorded per rung: `tokens`
(ligand heavy atoms are tokens the trunk pays for, and are nowhere in the residue count) and
`msa_rows_to_model` (counted after `_parse_a3m_to_msa` dedup, not off `>` records).
