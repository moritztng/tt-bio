#!/bin/bash
cd /home/moritz/.coworker/wt/protenix-v2-640aa-hang-char-pre
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-640aa-hang-characterize
exec /home/moritz/tt-bio/env/bin/python3 scripts/protenix_hang_probe.py   --trials 15 --card 0 --timeout 300 --gap 25   --out perf/hangprobe/pc_c0_pre_d5ade211
