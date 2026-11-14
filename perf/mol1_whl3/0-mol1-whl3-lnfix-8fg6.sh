# mol1-whl3 lnfix: 8fg6 s42 on 7fe328327 (59893ab61 + fp32_layer_norm on the Wormhole single-track Transition), served pLDDT + per-layer trunk s (WHL3_LAYERS) + head inputs
# mem: 60
# threads: 4
# cap: 3600
R=/home/ubuntu/spd/mol1-whl3; I=/home/ubuntu/spd/mol1-quality/inputs; T=$R/tree-7fe328327
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
d=$R/out/lnfix/.runs/8fg6-assembly1/seed_42; mkdir -p $d
TREE=$T bash $R/run_cell.sh $CHIP $d 2700 -- bash -c "export PYTHONPATH=$R/hook-tap:\$PYTHONPATH WHL3_ARMS=served WHL3_LAYERS=1 WHL3_DUMP=$R/out/dump/lnfix_8fg6-assembly1_s42; cd $I && python -c \"from tt_bio.main import cli; cli()\" predict 8fg6-assembly1.json --model mol1 --seed 42 --debug --out_dir $R/out/lnfix"
r=$?; echo rc=$r > $d/rc; exit $r
