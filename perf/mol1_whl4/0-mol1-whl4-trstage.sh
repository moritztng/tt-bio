# mol1-whl4: fp32_single_transition stage by stage vs float64 (same bf16 weights), 7fxn block 43 input, tree 773a80594; ~1 min
# mem: 10
# threads: 2
# cap: 600
R=/home/ubuntu/spd/mol1-whl4
export TREE=$R/tree-773a80594 TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=2
d=$R/out/trstage; mkdir -p $d
bash $R/run_cell.sh $CHIP $d 500 -- timeout 500 python $R/tr_stage_probe.py $R/x0_43_7fxn.pt $d/tr_stage.json
r=$?; echo rc=$r > $d/rc; exit $r
