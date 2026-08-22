#!/bin/bash
cd /home/ttuser/.coworker/wt/protenix-v2-640aa-hang-characterize
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-640aa-hang-characterize
exec python3 scripts/protenix_hang_probe.py --rung 640 --trials 10 --card 3   --timeout 900 --gap 25 --out perf/hangprobe/qb1_c3_long
