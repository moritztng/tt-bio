# mol1-whl3 WH pLDDT repro: 8fg6 + 8k83 s42 on main 59893ab61, confidence head rerun under lever ablations (hook/sitecustomize.py)
# mem: 60
# threads: 4
# cap: 7200
R=/home/ubuntu/spd/mol1-whl3; I=/home/ubuntu/spd/mol1-quality/inputs; T=/home/ubuntu/spd/mol1-wh1024cap/tree-59893ab61
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
rc=0
cell() { # case seed
  d=$R/out/main/.runs/$1/seed_$2; grep -qs rc=0 $d/rc && return 0; mkdir -p $d; rm -f $R/out/dump/$1_s$2/arms.part.pt
  TREE=$T bash $R/run_cell.sh $CHIP $d 5400 -- bash -c "export PYTHONPATH=$R/hook:\$PYTHONPATH WHL3_DUMP=$R/out/dump/$1_s$2; cd $I && python -c \"from tt_bio.main import cli; cli()\" predict $1.json --model mol1 --seed $2 --debug --out_dir $R/out/main"
  r=$?; echo rc=$r > $d/rc; [ $r = 0 ] || rc=$r
}
cell 8fg6-assembly1 42
cell 8k83-assembly1 42
exit $rc
