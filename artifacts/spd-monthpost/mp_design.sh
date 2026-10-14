#!/bin/bash
# mp_design.sh SHA9: overlay main's design-model harness (perf/spd/bench.py, perf/spd/bc2_census.py, bg_page.py) into
# qb1:~/spd/monthpost/trees/m<SHA9> (engine stays the tree's own tt_bio) and append the design cells to its jobs.txt.
# AF2-IG = bench.py --model af2ig on 3 PD-L1+binder-80 complexes x 5 warm (spd-gap-bind's GAP cell); BindCraft 2 =
# bc2_census.py 30 gradient calls on PD-L1 binder 80 (s/iteration); BoltzGen = bg_page.py, CLI, 6 designs, s/design.
set -eu
SHA=$1; R=/home/moritz/tt-bio; G=/home/ttuser/spd/monthpost; T=$G/trees/m$SHA; O=$G/out/$SHA
ssh -o BatchMode=yes qb1 "mkdir -p $T/perf/spd"
git -C $R show origin/main:perf/spd/bench.py | ssh -o BatchMode=yes qb1 "cat > $T/perf/spd/bench.py"
git -C $R show origin/main:perf/spd/bc2_census.py | ssh -o BatchMode=yes qb1 "cat > $T/perf/spd/bc2_census.py"
ssh -o BatchMode=yes qb1 "cp ~/perf-page-tt-latest/bg_page.py $T/perf/spd/bg_page_mp.py; cat >> $O/jobs.txt" <<J
af2ig-normal 1800 python perf/spd/bench.py --out $O/af2ig-normal --chip \$TT_VISIBLE_DEVICES --model af2ig --inputs af2ig_pdl1_l80_s0,af2ig_pdl1_l80_s1,af2ig_pdl1_l80_s2 --warm 5 --data /home/ttuser/spd/spd-gap-bind/data
bc2-normal 2700 JAPANFOLD_BC2_AF2_WEIGHTS=/home/ttuser/bcx_e2e/af2_params /home/ttuser/bcx_e2e_venv/bin/python perf/spd/bc2_census.py --out $O/bc2-normal --bc2 /home/ttuser/bcx_e2e/bc2 --binder 80 --seed 101 --at 31 --profile 0 --wrap census
boltzgen-normal 7200 PPL_OUT=$O/boltzgen-normal PPL_FAST=0 python perf/spd/bg_page_mp.py
boltzgen-fast 7200 PPL_OUT=$O/boltzgen-fast PPL_FAST=1 python perf/spd/bg_page_mp.py
J
ssh -o BatchMode=yes qb1 "tail -4 $O/jobs.txt | cut -c1-80; wc -l < $O/jobs.txt"
