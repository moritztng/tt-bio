# A wedge is visible in device telemetry a minute after it happens, in two counters

Two device rows wedged this campaign, and both were first misread off CPU%. The parent called one
**healthy at 100 % CPU while the chip had been stopped for ten minutes**. `pcpu` cannot work as a
discriminator here: tt-metal's completion-wait threads spin rather than block, so a host polling a
dead chip looks exactly like a host feeding a busy one.

qb2 already logs something that does work. `/home/ttuser/qbcard/cardtel.py` writes `cardtel.tsv`
every 2 s with per-card AI clock, board power, and NoC data-word counters. Two of them separate the
states cleanly, and because the history is on disk a wedge can be **dated** afterwards rather than
guessed at.

## The measurement, from `c10-size-scaling`'s 768 aa ladder on card 0

| state | `slv_rd_data_word_sent0` | `mst_rd` spread | board power |
|---|---|---|---|
| folding (00:52–00:59) | 323,861 – 4,410,682 /s | **100 %** | 27 – 105 W |
| wedged (01:00 onward) | **0, every sample** | **0.07 %** | 22 – 24 W |

`slv_rd` is the device answering reads; at zero it is serving nothing. `mst_rd` pinned to a constant
rate — 0.07 % spread across twelve samples — is a host poll loop at a fixed period. Together they
are *"a host busy-polling a chip that has stopped"*, which is exactly what the wedge was. The
transition is sharp: the last folding sample is 00:59:04 and `slv_rd` is 0 from 01:00:00.

Power separates too, but less sharply — the wedged band (22–24 W) sits just under the folding
minimum (27 W), so power alone is the weaker signal. The counters do not overlap at all.

## The second wedge cannot be scored this way, and that is worth knowing

`c10-fold-census`'s smoke wedged at ~01:28. From 01:29 `cardtel` read **zero** for clock, power and
both counters, while the fleet's own `aiclk_watch` — reading
`/sys/class/tenstorrent/tenstorrent!N/tt_aiclk` — kept reporting 800 MHz. Two readers disagreeing is
itself a finding. What it means here is that this discriminator was **not available** for that
incident, so the signature rests on one scored wedge, not two.

## The parsing trap, which produced a confident wrong answer first

`cardtel.tsv` spans **38.9 hours**. Filtering it by wall-clock `%H:%M:%S` silently matches the same
time on two different days, and the first version of this reduction did exactly that — it mixed an
idle stretch from the previous day into the fold window. **Use absolute epochs.** The tell was an
epoch of `1789520281` inside a window that should have held `1789606xxx`.

## Limits

One scored incident, not a validated detector. It has never been run against a false positive such
as a long CPU-side phase between folds, and this campaign has one of those on record —
`replay.py --only none` is a no-op control arm where an idle chip is *correct*. Detection is ~60 s,
not instant, because cardtel samples every 2 s and the rate needs a window. Nothing here reproduces
or explains either wedge; it dates one and says what to watch.

    python3 wedge_telemetry.py                      # writes wedge_telemetry.json
    python3 -m pytest test_wedge_telemetry.py -q    # 10 controls
    python3 extract.py                              # regenerates the numbers, run ON qb2
