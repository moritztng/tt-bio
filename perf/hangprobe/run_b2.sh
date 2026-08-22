#!/bin/bash
cd /home/ttuser/.coworker/wt/protenix-v2-640aa-hang-characterize
exec python3 scripts/protenix_hang_probe.py --trials 24 --card 3 --timeout 420 --gap 25   --out perf/hangprobe/qb1_c3_grid_default
