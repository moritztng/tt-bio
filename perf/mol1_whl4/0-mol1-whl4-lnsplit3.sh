# mol1-whl4: WH fp32 LN statistics, ttnn.mean vs two-term split, on 7fxn block 43's Transition input; ~1 min
# mem: 10
# threads: 2
# cap: 600
R=/home/ubuntu/spd/mol1-whl4
export TREE=$R/tree-dd267265b TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=2
d=$R/out/lnsplit3; mkdir -p $d
bash $R/run_cell.sh $CHIP $d 500 -- timeout 500 python $TREE/perf/mol1_whl4/ln_split_probe.py $R/x0_43_7fxn.pt $d/ln_split3.json
r=$?; echo rc=$r > $d/rc; exit $r
