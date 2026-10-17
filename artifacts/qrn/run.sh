#!/bin/bash
D=/home/moritz/.coworker/wt/spd-gap-whlm/artifacts/qrn
rsync -a --exclude '*.part' --exclude '*.run' ubuntu@172.16.102.108:spd/spd-gap-whlm/qrn/ $D/runs/ || { echo rsync-fail; exit 1; }
cd /home/moritz/.coworker/wt/spd-gap-whlm
nice -n 19 timeout 5400 /home/moritz/pfm-accuracy-data/venv/bin/python perf/spd/grade.py $D/runs/f0/* $D/runs/qr/* --base f0 --test qr --mode normal --jobs 2 --out $D/report.md
echo "end rc=$?"
