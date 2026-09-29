# bwx-perf sitting results

`hhost3` is the H-HOST sitting: where BindCraft 2's extra-MSA stack and multimer template
embedder should run on a Wormhole Galaxy chip, on card or left in BindCraft 2's own JAX on the
host, which is what `bindcraft2.predictor()` defaults to.

Eight arms alternated at the process boundary in one sitting on one chip, `a b b a a b b a`,
five rounds each and the first discarded as the compile round. 288 tokens (hPDL1 115 + binder
146), the shipped five-`multimer_v3` configuration, `--exact 0`, dev Galaxy .107
(UF-EV-A4-GWH01) chip 30 = `/dev/tenstorrent/6` at `0000:c7:00.0`, 2026-09-29 22:13-22:32Z,
`5eba2312a`. AICLK sampled at 1 Hz throughout: **1000 MHz median, 0 of 454 samples under it**,
which is a Wormhole chip's ceiling.

    both in JAX (the shipped default)  29.423 s a round   host 17.188 + device 12.294
    both on card                       16.267 s a round   host  2.417 + device 13.856
                                       1.8087x

`hhost3_report.json` is `perf/bwx_perf/report.py`'s output; `hhost3_events.tar.gz` holds each
arm's `round_events.json`, which carries the per-round meter events, the clock samples and the
full lever stamp, so the table can be recomputed rather than trusted.
