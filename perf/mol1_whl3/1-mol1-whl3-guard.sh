# mol1-whl3 WH s_trunk: 8fg6 s42 on main 59893ab61 with TT_BIO_DEST_GUARD=1 (served levers, and every lever off), inputs + per-layer trunk s dumped (WHL3_LAYERS), plus an unguarded layers arm and sfp32 (trunk single-track residual in fp32)
# mem: 60
# threads: 4
# cap: 7200
R=/home/ubuntu/spd/mol1-whl3; I=/home/ubuntu/spd/mol1-quality/inputs; T=/home/ubuntu/spd/mol1-wh1024cap/tree-59893ab61
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
rc=0
cell() { # arm case seed [env]
  d=$R/out/$1/.runs/$2/seed_$3; grep -qs rc=0 $d/rc && return 0; mkdir -p $d; rm -f $R/out/dump/$1_$2_s$3/arms.part.pt
  TREE=$T bash $R/run_cell.sh $CHIP $d 2700 -- bash -c "export PYTHONPATH=$R/hook:\$PYTHONPATH WHL3_ARMS=served WHL3_LAYERS=1 WHL3_DUMP=$R/out/dump/$1_$2_s$3 $4; cd $I && python -c \"from tt_bio.main import cli; cli()\" predict $2.json --model mol1 --seed $3 --debug --out_dir $R/out/$1"
  r=$?; echo rc=$r > $d/rc; [ $r = 0 ] || rc=$r
}
cell guard 8fg6-assembly1 42 TT_BIO_DEST_GUARD=1
cell layers 8fg6-assembly1 42
cell sfp32 8fg6-assembly1 42 WHL3_SFP32=1
cell guardnolev 8fg6-assembly1 42 "TT_BIO_DEST_GUARD=1 TT_BIO_LEVERS=none"
exit $rc
