# mol1-whl3 tap: 8fg6 s42 on main 59893ab61, the single Transition of trunk block 43 in recycle cycle 1 (PairformerLayer call 91), every intermediate dumped (WHL3_TAP) to grade each op against upstream fp32
# mem: 60
# threads: 4
# cap: 3600
R=/home/ubuntu/spd/mol1-whl3; I=/home/ubuntu/spd/mol1-quality/inputs; T=/home/ubuntu/spd/mol1-wh1024cap/tree-59893ab61
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
d=$R/out/tap/.runs/8fg6-assembly1/seed_42; mkdir -p $d
TREE=$T bash $R/run_cell.sh $CHIP $d 2700 -- bash -c "export PYTHONPATH=$R/hook-tap:\$PYTHONPATH WHL3_ARMS=served WHL3_TAP=91 WHL3_DUMP=$R/out/dump/tap_8fg6-assembly1_s42; cd $I && python -c \"from tt_bio.main import cli; cli()\" predict 8fg6-assembly1.json --model mol1 --seed 42 --debug --out_dir $R/out/tap"
r=$?; echo rc=$r > $d/rc; exit $r
