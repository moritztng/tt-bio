# Hardware telemetry and screens

The four chips of the QuietBox, made visible: power, temperature and what each chip is folding
(the clock is sampled and logged, not shown), plus the measured comparison and a dataflow screen for experts.

| file | what |
|------|------|
| `telemetry.py` | per-chip sampler, the fold ledger and `record_fold()` |
| `facts.py` | builds `web/app/lanes/facts.json` from `site/data/perf-512aa.json` |
| `serve.py` | stand-alone server for development and screenshots |
| `../web/app/lanes/` | the three screens: `lanes`, `compare`, `depth` |

```sh
python3 demo/sc26/hardware/serve.py     # http://127.0.0.1:8627/app/lanes/?view=lanes
python3 demo/sc26/hardware/telemetry.py # one snapshot as JSON
python3 -m pytest demo/sc26/hardware/
```

## Where the numbers come from

Everything is read from the files the Tenstorrent kernel driver publishes under
`/sys/class/tenstorrent/tenstorrent!N/` (`tt_aiclk`, `tt_heartbeat`, `tt_card_type`) and its hwmon
directory (power, ASIC temperature, core voltage). No device is opened and no `tt-smi` runs, so
sampling cannot disturb a fold. Four samples a second cost 0.5 % of one CPU core; one sample of all
four chips takes about 1 ms.

A chip that is not giving readings shows as `resetting`, never as a number: a dead ARC returns
0xFFFFFFFF, a stopped firmware stops its heartbeat, and a board reset removes the node for a
moment. `tt-smi` is used only when a driver is too old to publish `tt_aiclk`, then at most one at
a time, under a timeout, and its process group is reaped.

## The `chips` message

The engine's server returns `Monitor.snapshot()` at `GET /telemetry`; the screens poll it and take
it through `view.update(msg)`.

```json
{"type": "chips", "t": 1790985432.1, "source": "live", "rate_hz": 4.0, "sample_ms": 1.1,
 "chips": [{"card": 0, "node": 0, "bdf": "0000:01:00.0", "board": "p300c",
            "state": "folding", "aiclk_mhz": 1350, "power_w": 88.0, "power_max_w": 125.0,
            "temp_c": 63.2, "vcore_v": 0.82, "power_60s": [31.0, 64.0],
            "folding": {"model": "esmfold2", "name": "GFP", "residues": 238, "elapsed_s": 1.2},
            "folds_today": 12, "residues_per_s_today": 118.4,
            "last_fold": {"model": "esmfold2", "name": "GFP", "residues": 238, "seconds": 1.9,
                          "residues_per_s": 125.3,
                          "aiclk_during": {"median": 1350, "min": 1343, "n": 8}}}]}
```

`state` is `folding` (a demo fold is running), `idle`, `busy` (clock up, but no demo fold on it) or
`resetting`. Readings are `null` when the chip is resetting. `card` is the index
`TT_VISIBLE_DEVICES` uses, which is PCI order and not the `/dev/tenstorrent` node number.
`source` is `live` or the name of a recording, and the screen prints it.

## Telling the telemetry about folds

The engine's server records every fold it starts, finishes or loses; the ledger tails the file,
so folds survive a restart of either side.

```python
from telemetry import record_fold
record_fold("start", card, model="esmfold2", name="GFP", residues=238)
record_fold("done", card, seconds=1.91)    # or record_fold("fail", card)
```

The file is `~/.local/state/sc26/folds.jsonl` (override with `--events`). The AICLK shown beside
a finished fold is the median of the samples taken while it ran.

## The comparison

`facts.json` is generated, never edited: it holds the published 512-residue rows from
tt-bio.com/benchmarks with their board and date, and the BindCraft 2 round from
`docs/bindcraft2.md`. A test fails when either source changes and the file was not rebuilt.
Every row measured on all four platforms is shown, Protenix-v2 included: its weights' licence
restricts hosting it, not running it on your own card.
