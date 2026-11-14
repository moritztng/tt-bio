# mol1-whl3 ln probe 2: ttnn.layer_norm on trunk block 43's captured Transition input (tap.pt) composed from fp32 primitives, tree 59893ab61; ~1 min
# mem: 10
# threads: 2
# cap: 900
R=/home/ubuntu/spd/mol1-whl3; T=/home/ubuntu/spd/mol1-wh1024cap/tree-59893ab61; D=$R/out/dump/tap_8fg6-assembly1_s42
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=2
d=$R/out/lnprobe2; mkdir -p $d
TREE=$T bash $R/run_cell.sh $CHIP $d 600 -- timeout 600 python $R/ln_probe2.py $D/tap.pt $D/lnw43.pt $d/ln_probe2.json
r=$?; echo rc=$r > $d/rc; exit $r
