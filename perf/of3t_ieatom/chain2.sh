#!/usr/bin/env bash
# of3t-ieatom chain 2 on qb2 card 3. wk/of3t 59646c2bb carries of3t-composed64's compose defect
# (pair transitions dropped under the tape; IE64B reproduces its CM64 loss 7.386313018564488), so
# the score runs on PW64F's own tree f89f871ce with this row's commits on top (scoretree), and
# f89f871ce itself runs on card 3 as the A/A control against PW64F (card 2).
W=/home/ttuser/.coworker/wt/of3t-ieatom; S=/home/ttuser/of3t_ieatom; L=$S/chain2.log
T=$S/scoretree; P=$S/scoreprefix; B=$S/pwtree
cd $W
[ -d $P ] || git worktree add -q --detach $P "$(git -C $T rev-parse HEAD~2)"
[ -d $B ] || git worktree add -q --detach $B f89f871ce
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][3]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain2.txt 2>&1 &
M=$!
source /home/ttuser/tt-bio-dev/env/bin/activate
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
ENV=(TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-ieatom OMP_NUM_THREADS=8)
for X in prefix_pw:$P fix_pw:$T; do
  n=${X%%:*}; tree=${X#*:}
  echo "=== control $n start $(date -u +%FT%TZ) $(git -C $tree rev-parse --short HEAD)" >> $L
  (cd $tree && env "${ENV[@]}" PYTHONPATH=$tree timeout 900 python3 $W/perf/of3t_ieatom/guard_control.py \
     --batch /home/ttuser/of3t_fullstep64/batch_step003_t64.pt --out $W/perf/of3t_ieatom/GUARD_CONTROL_${n^^}.json 2>&1 \
     | grep -vE 'TT_FATAL|DEBUG|Config\{| info |^\s*$|===|Total nodes|Degree hist' | tail -5) >> $L
  echo "=== control $n exit $(date -u +%FT%TZ)" >> $L
done
bash perf/of3t_ieatom/devarm.sh PF64 $T --weights-out $S/weights_walked_PF.pt >> $L 2>&1
bash perf/of3t_ieatom/devarm.sh PW64C3 $B >> $L 2>&1
kill $M
echo "=== chain done $(date -u +%FT%TZ)" >> $L
