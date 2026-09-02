# protenix-v2 size ceiling ladder (Wormhole)

Rungs at 128, 256, 512, 980, 1024, 1056, 1088, 1120 and 1152 residues, built only from
`examples/abag_xm` chains whose MSAs are already in the shared cache, so every rung folds
`--msa_cache_only` with a real alignment instead of single-sequence. A fixed core of 9q7y chain 1
(629 aa) + 9qqe chain 1 (268 aa) = 897, plus further chains that set the rung.

`px1120s.yaml` and `px1152s.yaml` are the matched versions: the px1024 complex plus one more
chain, so they differ from the passing 1024 rung by token count and nothing else. The plain
`px1120.yaml` is not matched — its third chain carries a 14 643-row alignment, and it dies on the
MSA tensor rather than on the token axis. Both 1120s fail, for different reasons.

`run_one_chip.sh` runs a list of `<tree>:<rung>` jobs sequentially on one chip and never claims a
second. `run_ladder.sh` is the multi-chip version; prefer the single-chip one unless you own the
chips, because two dispatchers will put two folds on one card and the device lease will refuse the
second.

    ./run_one_chip.sh "fix:1024 fix:1120s"

Chip numbering: `TT_VISIBLE_DEVICES` is a UMD id, which on the JapanFold Galaxy is not the `/dev`
node. UMD enumerates by PCI BDF, giving `u < 16 -> node u+16`, `16 <= u < 24 -> u-8`,
`u >= 24 -> u-24`. `/dev/tenstorrent/5` is `TT_VISIBLE_DEVICES=29`; `TT_VISIBLE_DEVICES=5` is
`/dev/tenstorrent/21`.

Measured 2026-09-02, warm MSA, platform flags. Before the OuterProductMean byte gate, 980 and 1024
both throw on the same 2 147 483 648 B request. After it, 1088 folds in 1013 s (twice,
byte-identical on two chips) and 1120 is the first failure. `results.jsonl` has every run. See
`~/.coworker/state/ceiling-protenix-v2.md`.
