#!/usr/bin/env bash
# of3t-ieatom chain 3 on qb2 card 3, fp32 encoder: s_input probe, guard control on the fix
# (score tree and branch head), arm PF64B on PW64F's tree plus this row, walked weights dumped.
W=/home/ttuser/.coworker/wt/of3t-ieatom; S=/home/ttuser/of3t_ieatom; L=$S/chain3.log; T=$S/scoretree
cd $W
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][3]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain3.txt 2>&1 &
M=$!
source /home/ttuser/tt-bio-dev/env/bin/activate
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
ENV=(TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-ieatom OMP_NUM_THREADS=8)
F='TT_FATAL|DEBUG|Config\{| info |^\s*$|===|Total nodes|Degree hist'
echo "=== probe start $(date -u +%FT%TZ) $(git -C $T rev-parse --short HEAD)" >> $L
(cd $T && env "${ENV[@]}" PYTHONPATH=$T timeout 900 python3 $W/perf/of3t_ieatom/probe_sinput.py \
   --batch /home/ttuser/of3t_fullstep64/batch_step003_t64.pt --out $S/SINPUT_PF64B.pt 2>&1 | grep -vE "$F" | tail -3) >> $L
for X in fix_pw2:$T fix_branch:$W; do
  n=${X%%:*}; tree=${X#*:}
  echo "=== control $n start $(date -u +%FT%TZ) $(git -C $tree rev-parse --short HEAD)" >> $L
  (cd $tree && env "${ENV[@]}" PYTHONPATH=$tree timeout 900 python3 $W/perf/of3t_ieatom/guard_control.py \
     --batch /home/ttuser/of3t_fullstep64/batch_step003_t64.pt --out $W/perf/of3t_ieatom/GUARD_CONTROL_${n^^}.json 2>&1 \
     | grep -vE "$F" | tail -5) >> $L
done
bash perf/of3t_ieatom/devarm.sh PF64B $T --weights-out $S/weights_walked_PFB.pt >> $L 2>&1
kill $M
echo "=== chain done $(date -u +%FT%TZ)" >> $L
