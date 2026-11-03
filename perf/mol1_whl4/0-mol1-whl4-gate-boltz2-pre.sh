# mol1-whl4 Boltz-2 512 WH smoke, pre-whl3 59893ab61: bfb187307 moves Boltz-2's Transition back off fp32_single_transition,
# so its fold should match this tree's (gate-boltz2-512/fix digest d0900d852b6635f3).
# mem: 40
# threads: 16
# cap: 1800
R=/home/ubuntu/spd/mol1-whl4; P=$R/tree-59893ab61; F=$R/tree-bfb187307
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=16
unset TT_BIO_LEVERS TT_BIO_DRAM_PEAK
o=$R/out/gate-boltz2-pre; mkdir -p $o/pre
TREE=$P bash $R/run_cell.sh $CHIP $o 1700 -- python $F/perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree $P --size 512 --repeat 1 --label pre --out $o/pre/result.json
r=$?; echo rc=$r > $o/rc; exit $r
