# protenix-v2 size ceiling ladder (Wormhole)

Four rungs at 980, 1024, 1056 and 1088 residues, built only from `examples/abag_xm` chains whose
MSAs are already in the shared cache, so every rung folds `--msa_cache_only` with a real alignment
instead of single-sequence. A fixed core of 9q7y chain 1 (629 aa) + 9qqe chain 1 (268 aa) = 897,
plus one third chain that sets the rung.

`run_ladder.sh` claims one chip per rung on the JapanFold Galaxy. It only ever takes chips outside
the production pool, and re-checks `sudo lsof /dev/tenstorrent/*` twice, 15 s apart, before each
claim. Chip numbering: `TT_VISIBLE_DEVICES` is a UMD id, which on this box is not the `/dev`
node — UMD enumerates by PCI BDF, giving `u < 16 -> node u+16`, `16 <= u < 24 -> u-8`,
`u >= 24 -> u-24`.

    ./run_ladder.sh "1024 1056 980 1088"

Results land in `results.jsonl` beside the script, one line per rung.

Measured 2026-09-02 on tree 46613cec: 980, 1024 and 1056 all OOM in the trunk. 980 and 1024 ask
for the same 2 147 483 648 B buffer, because the failing allocation pads its token axis to a
multiple of 64 and both round to 1024. The throw is fragmentation, not exhaustion: 331 MB free per
bank against a 179 MB request, largest free block 136 MB. See
`~/.coworker/state/ceiling-protenix-v2.md`.
