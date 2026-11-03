# mol1-whl4 merge gate: boltz2 512 on WH, base 974d7c9ee (main merge-base) vs fix bfb187307 (fp32 trunk single track + host
# confidence single track), one chip, one process each, smoke: cold + 1 warm per arm, base then fix (Boltz-2 WH Transition goes back off fp32_single_transition); AICLK in aiclk.tsv.
# mem: 40
# threads: 16
# cap: 5400
R=/home/ubuntu/spd/mol1-whl4; B=$R/tree-974d7c9ee; F=$R/tree-bfb187307
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=16
unset TT_BIO_LEVERS TT_BIO_DRAM_PEAK
o=$R/out/gate-boltz2-512; mkdir -p $o/base $o/fix $o/base2
TREE=$F bash $R/run_cell.sh $CHIP $o 5100 -- bash -c "python $F/perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree $B --size 512 --repeat 1 --label base --out $o/base/result.json && python $F/perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree $F --size 512 --repeat 1 --label fix --out $o/fix/result.json "
r=$?; echo rc=$r > $o/rc; exit $r
