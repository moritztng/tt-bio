#!/usr/bin/env bash
# of3t-ieatom chain 5 on qb2 card 3, after the D264 fixes (direction change, pass 426):
#   CM64F  base 59646c2bb + the two D264 commits, no D263: must be bit-identical to PW64F
#   guard  checkpoint-side guard on the fixed tree (base + D264 + D263): must stay silent
#   PF64F  base + D264 + D263, walked weights for the bijection
#   infab  OpenFold3 fold digest base vs fixed tree, census inside the folding process
W=/home/ttuser/.coworker/wt/of3t-ieatom; S=/home/ttuser/of3t_ieatom; L=$S/chain5.log
CM=$S/cmtree; PF=$S/pftree
cd $W
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][3]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain5.txt 2>&1 &
M=$!
source /home/ttuser/tt-bio-dev/env/bin/activate
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
ENV=(TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-ieatom OMP_NUM_THREADS=8)
F='TT_FATAL|DEBUG|Config\{| info |^\s*$|===|Total nodes|Degree hist'
bash perf/of3t_ieatom/devarm.sh CM64F $CM >> $L 2>&1
echo "=== control fix_pf start $(date -u +%FT%TZ) $(git -C $PF rev-parse --short HEAD)" >> $L
(cd $PF && env "${ENV[@]}" PYTHONPATH=$PF timeout 900 python3 $W/perf/of3t_ieatom/guard_control.py \
   --batch /home/ttuser/of3t_fullstep64/batch_step003_t64.pt --out $W/perf/of3t_ieatom/GUARD_CONTROL_FIX_PF.json 2>&1 \
   | grep -vE "$F" | tail -5) >> $L
bash perf/of3t_ieatom/devarm.sh PF64F $PF --weights-out $S/weights_walked_PFF.pt >> $L 2>&1
echo "=== infab start $(date -u +%FT%TZ)" >> $L
(cd $PF && timeout 2400 python3 $W/perf/of3t_ieatom/infab.py --before $S/base --after $PF --card 3 \
   --workdir $S/infab --models openfold3 --holder worker:of3t-ieatom \
   --out $W/perf/of3t_ieatom/INFAB.json 2>&1 | grep -E "INFAB|Error|Traceback" ) >> $L
kill $M
echo "=== chain done $(date -u +%FT%TZ)" >> $L
