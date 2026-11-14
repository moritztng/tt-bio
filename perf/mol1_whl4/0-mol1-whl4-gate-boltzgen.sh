# mol1-whl4 BoltzGen WH smoke on bfb187307: its Transitions go back off fp32_single_transition; one design, design step.
# mem: 60
# threads: 8
# cap: 2400
R=/home/ubuntu/spd/mol1-whl4; F=$R/tree-bfb187307
export TREE=$F TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=8
unset TT_BIO_LEVERS TT_BIO_DRAM_PEAK
o=$R/out/gate-boltzgen; mkdir -p $o
bash $R/run_cell.sh $CHIP $o 2300 -- python -c "from tt_bio.main import cli; cli()" design $F/examples/binder.yaml --model boltzgen --num_designs 1 --steps design --out_dir $o/design
r=$?; echo rc=$r > $o/rc; exit $r
