# mol1-whl3 lnpre: 8k83 s42 on 7fe328327 + fp32 pre_norm_s on WH (scratch tree), served pLDDT + per-layer trunk s (WHL3_LAYERS) + head inputs
# mem: 60
# threads: 4
# cap: 3600
R=/home/ubuntu/spd/mol1-whl3; I=/home/ubuntu/spd/mol1-quality/inputs; T=$R/tree-prenorm-7fe328327
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
d=$R/out/lnpre/.runs/8k83-assembly1/seed_42; mkdir -p $d
TREE=$T bash $R/run_cell.sh $CHIP $d 2700 -- bash -c "export PYTHONPATH=$R/hook-tap:\$PYTHONPATH WHL3_ARMS=served WHL3_LAYERS=1 WHL3_DUMP=$R/out/dump/lnpre_8k83-assembly1_s42; cd $I && python -c \"from tt_bio.main import cli; cli()\" predict 8k83-assembly1.json --model mol1 --seed 42 --debug --out_dir $R/out/lnpre"
r=$?; echo rc=$r > $d/rc; exit $r
