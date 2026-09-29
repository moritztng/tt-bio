# BindCraft 2 on a Wormhole Galaxy chip

The first BindCraft 2 gradient rounds to run on Wormhole. Dev Galaxy `.107` (`UF-EV-A4-GWH01`,
Osaka dev box, not production), **chip 30** = sysfs node 6, PCI `0000:c7:00.0`, `Arch.WORMHOLE_B0`,
compute grid **8x9**, so `tt_bio.tenstorrent._IS_SMALL_GRID` is the path in force. tt-bio
`2d174fd5b`, ttnn 0.68.0 (`cp312` wheel), BindCraft 2 `7a2dfdb`, AF2 parameters 2022-12-06.
The chip was borrowed through tt-bio's own lease in the agent's lease dir, with the dev agent
stopped for about 8 s per launch and restarted, so the other 30 chips served throughout.

`out/*.json` are the harness's own event logs, one per arm. Each carries its AICLK samples,
taken at 1 Hz **during** the run from the chip's sysfs; every reading below is at a median of
**1000 MHz**, which is the Wormhole ceiling.

| arm | tokens | configuration | warm round | device | host |
|---|---|---|---|---|---|
| `comp11_b146` | 288 | composed, extra-MSA and template on card | **15.72 s** | 12.71 s | 3.01 s |
| `comp_b146` | 288 | composed, extra-MSA and template in JAX | 26.62 s | 11.77 s | 14.85 s |
| `grad_b60` | 175 | `exact=1` tape (not what tt-bio ships) | 347.2 s (cold) | 324.8 s | |

Blackhole ran the first of those at **11.206 s** (host 1.670, device 9.470) on pc's p150a at
AICLK 1350, `bcx-p10-hostcut` in `state/perf10/bcx-BOARD.md`. So Wormhole costs **1.40x on the
round and 1.34x on the device column** at the same token axis and the same configuration, against
2.27-2.38x for folding at matched input.

Every fused kernel BindCraft 2 ships serves on the 8x9 grid, with nothing declined for a
Wormhole reason: `fused_hifi` 1248 served / 0 declined, `triatt_bw` 416 / 0, `rne_add` 5424 / 48
(the 48 are a shape mismatch that is not board-specific). `mm_layout`'s declines are all shape
reasons it gives on Blackhole too.

Two traps cost a pass each and are worth knowing before you re-run this:

- `perf/bcx_round/run_round.py --exact` defaults to 1 and its help calls that origin/main's
  default. It is not: `bindcraft2.predictor` ships `exact=False`, and the exact tape is 24.87x
  the round. A `--exact 1` reading is not a Wormhole regression, it is the slow tape on any board.
- `--exact 0` alone aborts at the first round boundary with `lever went inert mid-arm`, because
  `exact=False` makes `bindcraft2.fast_round` arm six levers the harness was not told to expect.
  Ask for them: `--triatt-hifi 1 --rne-kernel 1 --triatt-bw 1` with `TT_BIO_MM_LAYOUT=1
  TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1`.
