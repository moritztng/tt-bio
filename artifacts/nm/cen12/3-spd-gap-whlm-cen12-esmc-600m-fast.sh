[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 48
# cap: 1800
# threads: 8
# cen12: per-op census on the READY head (c8 + FFN LN L1)
O=/home/ubuntu/spd/spd-gap-whlm/cen12/esmc-600m-fast
[ -s $O/census.json ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255 || exit 3
bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1500 python perf/wh-embed/op_census.py --model esmc-600m --fast --n-seqs 7 --out $O/census.part.json && mv $O/census.part.json $O/census.json
