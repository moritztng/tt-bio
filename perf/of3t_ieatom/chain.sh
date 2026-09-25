#!/usr/bin/env bash
# of3t-ieatom chain on qb2 card 3: guard controls (pre-fix, fix), fix arm IE64F with a walked-
# weights dump for the bijection, base arm IE64B on the wk/of3t head the row is based on.
W=/home/ttuser/.coworker/wt/of3t-ieatom; S=/home/ttuser/of3t_ieatom; L=$S/chain.log
cd $W
[ -d $S/base ] || git worktree add -q --detach $S/base 59646c2bb
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][3]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain.txt 2>&1 &
M=$!
source /home/ttuser/tt-bio-dev/env/bin/activate
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
ENV=(TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-ieatom OMP_NUM_THREADS=8)
for T in prefix:$S/prefix fix:$W; do
  n=${T%%:*}; tree=${T#*:}
  echo "=== control $n start $(date -u +%FT%TZ) $(git -C $tree rev-parse --short HEAD)" >> $L
  (cd $tree && env "${ENV[@]}" PYTHONPATH=$tree timeout 900 python3 $W/perf/of3t_ieatom/guard_control.py \
     --batch /home/ttuser/of3t_fullstep64/batch_step003_t64.pt --out $W/perf/of3t_ieatom/GUARD_CONTROL_${n^^}.json 2>&1 \
     | grep -vE 'TT_FATAL|DEBUG|Config\{| info |^\s*$|===|Total nodes|Degree hist' | tail -5) >> $L
  echo "=== control $n exit $(date -u +%FT%TZ)" >> $L
done
bash perf/of3t_ieatom/devarm.sh IE64F $W --weights-out $S/weights_walked_IE.pt >> $L 2>&1
bash perf/of3t_ieatom/devarm.sh IE64B $S/base >> $L 2>&1
kill $M
echo "=== chain done $(date -u +%FT%TZ)" >> $L
