#!/usr/bin/env bash
# of3t-ieatom chain 4 on qb2 card 3: s_input by value from the host leg, gradient through the
# fp32 device encoder. Guard control on the score tree, then arm PF64C with walked weights.
W=/home/ttuser/.coworker/wt/of3t-ieatom; S=/home/ttuser/of3t_ieatom; L=$S/chain4.log; T=$S/scoretree
cd $W
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][3]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain4.txt 2>&1 &
M=$!
source /home/ttuser/tt-bio-dev/env/bin/activate
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
ENV=(TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-ieatom OMP_NUM_THREADS=8)
F='TT_FATAL|DEBUG|Config\{| info |^\s*$|===|Total nodes|Degree hist'
echo "=== control fix_pw3 start $(date -u +%FT%TZ) $(git -C $T rev-parse --short HEAD)" >> $L
(cd $T && env "${ENV[@]}" PYTHONPATH=$T timeout 900 python3 $W/perf/of3t_ieatom/guard_control.py \
   --batch /home/ttuser/of3t_fullstep64/batch_step003_t64.pt --out $W/perf/of3t_ieatom/GUARD_CONTROL_FIX_PW3.json 2>&1 \
   | grep -vE "$F" | tail -5) >> $L
bash perf/of3t_ieatom/devarm.sh PF64C $T --weights-out $S/weights_walked_PFC.pt >> $L 2>&1
kill $M
echo "=== chain done $(date -u +%FT%TZ)" >> $L
