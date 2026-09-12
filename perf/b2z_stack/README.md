# b2z-bh-stack-config — the platform layer under the 2.34x

Measured on qb2 card 3 (p300c Blackhole, one chip of board `...410D`), tt-metal wheel 0.68.0,
KMD 2.11.0, firmware bundle 19.11.0.

The campaign scores Boltz-2 512 aa against two roofs, 429.9 GB/s streaming and 85.96 TFLOP/s dense
bf16 HiFi4, and finds a uniform 2.34x deficit against both at once. A uniform multiplier across four
unrelated sub-units is the signature of a platform-wide effect, so this row checks the layer under
the model: the clock the part actually holds, the roofs re-measured on the card in hand, and the
number of Tensix cores the model is handed.

## Scripts

| file | what it does |
|---|---|
| `clk_sampler.sh` | AICLK, power, current and temperature for all four chips at 20 Hz, read straight from `/sys/class/tenstorrent` and hwmon. No device open, so it cannot perturb or wedge the run it measures. |
| `fold_clk.py` | A 512 aa Boltz-2 fold that records each fold's epoch start and end, so a clock trace can be sliced to the fold instead of the process. `B2Z_GRID_X` forces the compute grid width. |
| `clk_report.py` | Slices a trace against a fold record: per-fold min/p05/median/max AICLK, power, temperature, and the clock histogram. |
| `dispatch_probe.py` | Opens the device with WORKER or ETH dispatch and reports the compute grid each one yields. |
| `make_metalroot.sh` | Builds a shadow `TT_METAL_RUNTIME_ROOT` (symlinks to the installed wheel, one patched core descriptor) so a dispatch experiment touches no shared checkout. |
| `cmp_roofs.py` | Card-3 roofs against the campaign's card-2 roofs, ratio per shape and fidelity. |

## Reproducing

```
bash perf/b2z_stack/clk_sampler.sh out/clk.csv 20 &
TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:<you> \
  TT_MESH_GRAPH_DESC_PATH=$(python3 -c 'import ttnn,pathlib;print(pathlib.Path(ttnn.__file__).parent/"tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")') \
  benchlock.sh <you> -- python3 perf/b2z_stack/fold_clk.py out/fold.json
python3 perf/b2z_stack/clk_report.py out/clk.csv out/fold.json 3
```

`TT_MESH_GRAPH_DESC_PATH` is required whenever a p300c is pinned to one of its two chips: UMD sees
a one-chip p300 board, falls back to a CUSTOM cluster type, and aborts the open without a
descriptor. tt-bio's own CLI sets this for you; a bare script does not.
