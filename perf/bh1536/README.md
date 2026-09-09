# The 1536-token ladder on Blackhole

Can every structure and affinity model fold 1536 tokens on one Blackhole p150a without an OOM?
`size_limits.py` carries no Blackhole row for any model, which is honest but empty: nobody has
walked a ladder on this silicon. This directory is that walk.

`run_rung.py` runs one (model, size) rung and judges it on the artifact. A CIF has to exist and
carry the expected number of residues; for `nesso1` a `*_affinity.json` has to carry a number.
Exit status and the engine's own `"status": "ok"` field are both ignored, because both have
shipped green over an empty output folder before.

The fixture is the tandem-repeated CDK2 chain from `perf/size512/build_sweep_fixtures.py`, cut
to each rung, alignment depth held at 35 rows. Rungs are 1024 (the size already proven on
Wormhole, so a failure there is the card and not the size), 1152, 1300, 1408, 1536. 1300 is
there on purpose: it is not a multiple of 32, and the sizes between rungs are where a padding
bug hides.

The affinity fixtures add MTX to the same chain. That makes 1536 residues into 1569 tokens,
another count that is not a multiple of 32, on the one model whose ligand token axis has never
been censused.

    ./chain.sh boltz2:1536 rf3:1536 ...        # skips a rung already in results.jsonl

Results land in `results.jsonl` (one object per rung) and `sweep.log` (one line). Wall clock
here is NOT a clean benchmark: five tasks shared these boxes the night it was measured.
