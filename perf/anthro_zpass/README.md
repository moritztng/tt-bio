# Z-passes, the transpose class, and the pair-row axis — one capture, three questions

One instrumented 512 aa fold on qb2 card 1 (Blackhole), no timing taken, answers all three
questions `anthro-kernel-read` handed up. Full write-up in `~/.coworker/state/anthro-zpass-census.md`.

| script | question |
|---|---|
| `capture.py` | the fold. Arms `perf/b2z2_byte_floor/trace_block.py`'s buffer-keyed recorder on one trunk `PairformerLayer` call, and records every `ttnn.transpose` / `ttnn.permute` in the whole fold |
| `zpass.py` | Q1. z-sized passes per sub-unit, against Anthropic's 3-4 (attention epilogue) and 14 over 5 launches (trimul) |
| `tsplit.py` | Q2. the 0.5175 s `TransposeDeviceOperation` class split into addressing and data movement |
| `pairrow.py` | Q3. whether the pair-row reuse axis is expressible in the shipped SDPA path, and the price if it is not |

Reproduce:

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:anthro-zpass-census \
      python3 perf/anthro_zpass/capture.py --size 512 --block-call 8 --out-dir perf/anthro_zpass/out/c1
    python3 perf/anthro_zpass/zpass.py --json perf/anthro_zpass/out/c1/zpass_512.json
    python3 perf/anthro_zpass/tsplit.py
    python3 perf/anthro_zpass/pairrow.py

`out/c1/` holds the committed capture: `block_512.json.gz` (one PairformerLayer, 245 ops over 166
buffers), `transposes_512.json.gz` (12,702 calls), `zpass_512.json`, `manifest.json`. The fold ran
to pLDDT 0.845919 both times the capture was taken, and the transpose census reproduced call for
call, so the instrumentation does not move the model.

Two counts, three answers:

* **91.99 z units of DRAM traffic per PairformerLayer over 135 programs** (6.1731 GB). Triangle
  attention's epilogue is 3.00 units on the starting node and 5.00 on the ending node, against
  Anthropic's fused 3-4 and the stock path's 12-14. Triangle multiplication is 24.02 units over 12
  launches against their 14 over 5.
* **Zero of the fold's 12,702 transposes swap two untiled axes**, so none of the 0.5175 s class is
  the addressing form their ending-node kernel exploits. 73.55 % of the transposed bytes are the
  mixed form, where one token axis is tiled and the other is not.
* The pair-row axis is already expressed at R = 5 rows per core and is not register-capped, but the
  bias is re-read per pair row inside the stock SDPA reader, which is a kernel change and not a
  config. Priced, unmeasured, falsifier named in `pairrow.py`.
