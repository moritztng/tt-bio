#!/bin/bash
# waits for the card 0 booking (book.py owner rfd4-opendde-drift, 22:13Z), then folds 256 and 1024 on A/B/C/D
cd /home/ttuser/.coworker/wt/rfd4-opendde-drift
while [ $(date +%s) -lt 1792793585 ]; do sleep 30; done
python3 ~/spd/feed/book.py add rfd4-opendde-drift 0 4200 --note "opendde-drift A/B/C/D at 256,1024" >/dev/null
exec timeout 4100 python3 perf/rfd4/opendde-drift/ab.py --card 0 --rungs 256,1024 --reps 2 --out /home/ttuser/.coworker/wt/rfd4-opendde-drift/perf/rfd4/opendde-drift/ab A=/home/ttuser/scratch/shipflow/trees/f9b8ca9e9f9d B=/home/ttuser/scratch/opendde-drift/B C=/home/ttuser/scratch/opendde-drift/C D=/home/ttuser/scratch/shipflow/trees/437a234135fc
