#!/usr/bin/env bash
# of3t-paez step 1: the 384 device step forward only, dumping the confidence head inputs/outputs.
W=/home/ttuser/.coworker/wt/of3t-paez; S=/home/ttuser/of3t_paez
cd $W; source /home/ttuser/tt-bio-dev/env/bin/activate
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"device_info\"][1][\"telemetry\"][\"aiclk\"].strip())")"; sleep 5; done) > $S/aiclk_dump384b.txt 2>&1 &
M=$!
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
echo "=== start $(date -u +%FT%TZ) $(git rev-parse --short HEAD)"
TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:of3t-paez OMP_NUM_THREADS=8 PYTHONPATH=$W \
 timeout 3000 python3 perf/of3t_paez/devstep_out.py --outputs-out $S/out_D384b.pt --repr-out $S/repr_D384b.pt --conf-dump $S/conf_D384b.pt --forward-only \
 --denoise --exact on --draws /home/ttuser/of3t_fullstep64/draws.pt --grad-out $S/unused.pt --out $S/DEV_D384b.json 2>&1 | grep --line-buffered -vE "TT_FATAL|DEBUG|Config\{" | tail -30
echo "=== exit ${PIPESTATUS[0]} $(date -u +%FT%TZ)"
kill $M
