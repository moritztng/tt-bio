# Trace-region census

Bytes per DRAM bank each ttnn trace capture in tt-bio occupies, read from
`ttnn.get_memory_view(dev, TRACE)` after `end_trace_capture` by `hook/sitecustomize.py`.
Wormhole runs: whglx (j10glx02, 12 banks of 1073741792 B), `chain.py`, records in
`census_wh/`. Blackhole runs: pc p150a (8 banks), `bh_census.sh`, records in `census_bh/`.
All at a 256 MiB region, so no capture was truncated.

| capture (TRACE_REGIONS key) | site | run | WH B/bank | BH B/bank |
|---|---|---|---|---|
| diffusion | tenstorrent.py `_capture_diff_trace` | boltz2 cdk2x2 512 | 663552 | 1638400 |
| diffusion | | boltz2 cdk2x2 1536 | 671744 | 1728512 |
| diffusion | | boltzgen bg400 / bg1300 | 663552 / 679936 | |
| protenix | protenix.py `_capture_trace` | protenix-v2 512 | 868352 | 2154496 |
| protenix | | protenix-v2 1536 | refused by WH size limit | 2301952 |
| protenix | | protenix-v1 512 / opendde 512 | 868352 / 843776 | |
| esmc | esmc.py `_capture_esmc_trace` | one trace at 510 / 1534 | 663552 / 1245184 | |
| esmc | | 8 live traces, 1310-1534 (`inputs/esmc_fill8.fasta`) | 9625600 total | 8257536 total |
| rfd3 | rfd3/model.py decoder, 2 traces | binder example (120 res) / 500 res | 286720 / 294912 | 630784 |

A capture is its command stream, not its tensors, so the bytes barely move with size: 1.2%
from 512 to 1536 tokens for boltz2 on Wormhole, 7% for protenix-v2 on Blackhole, 3% from 120 to
500 residues for rfd3.

The table's unit is NOT bytes per bank. ttnn carves `trace_region_size` off every bank, but
`end_trace_capture` compares the device's live trace buffers summed over all banks against it.
A 2 MiB region, the per-bank reading doubled, refused boltz2's step: "Creating trace buffers of
size 7929856B on MeshDevice 3, but only 2097152B is allocated for trace region", a total that
lands as 663552 B on each of 12 banks. So `TRACE_REGIONS` is the largest reading x banks, doubled,
rounded up to a MiB (WH / BH): diffusion 16 / 27 MiB, protenix 20 / 36, esmc 221 / 126,
rfd3 7 / 10. What the chip loses is that figure on every bank, e.g. 192 MiB of a Wormhole chip
for the diffusion trace, against 3 GiB at the old 256 MiB.

## What happens at the edges

- Region too small (`open_probe.py 4096 <n_adds>`, pc BH): `end_trace_capture` raises
  `TT_FATAL ... get_trace_buffers_size() <= trace_region_size`, and an eager add on the same
  device afterwards returns the right value. Undersizing fails loudly and leaves the chip usable.
  ESMC catches it and runs eager; the diffusion, protenix and rfd3 captures fail the job.
- Region at a bank (`open_probe.py 1073741824`, whglx card 22): the DRAM view right after open
  reads `total_bytes_per_bank = 18446744073709551584` (2^64 - 32). The 32x32 add enqueues and
  `synchronize_device` never returns; SIGINT at 60 s did not end it. 256 MiB and 512 MiB on the
  same card, minutes earlier, opened, captured and replayed in under 2 s
  (`results/open_c22_*` on whglx). The stall is the region size, not a displaced carve-out.
  Card 22 is fenced (`state/cardblock-whglx-22`) until the glx_reset window.
- Region displacing the model: rfd3 at 1280 residues (`inputs/rfd3_1280.json`) with
  `RFD3_TRACE_DECODER=1` ran out of DRAM (`bank_manager.cpp:439`) before any capture under the
  old 256 MiB region, which takes 3 GiB of a 12 GiB Wormhole chip.
