# mol1-whl4 OpenDDE 512 WH at diffusion seed 1 (the gate cell runs seed 0): base 974d7c9ee vs fix bfb187307, cold + 1 warm,
# for the structure delta against this fixture's seed floor (whl3 Step 10: 3.5-3.7 A).
# mem: 40
# threads: 16
# cap: 3600
R=/home/ubuntu/spd/mol1-whl4; B=$R/tree-974d7c9ee; F=$R/tree-bfb187307; X=$F/perf/of3_4xpd/xmodel_seed.py
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=16
unset TT_BIO_LEVERS TT_BIO_DRAM_PEAK
o=$R/out/gate-opendde-seed1; mkdir -p $o/base $o/fix
TREE=$F bash $R/run_cell.sh $CHIP $o 3400 -- bash -c "python $X 1 --model opendde --tree $B --size 512 --repeat 1 --label base --out $o/base/result.json && python $X 1 --model opendde --tree $F --size 512 --repeat 1 --label fix --out $o/fix/result.json"
r=$?; echo rc=$r > $o/rc; exit $r
