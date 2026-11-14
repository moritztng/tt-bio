# mol1-whl3 WH pLDDT repro: 8fg6 + 8k83 s42 on main 59893ab61 (+ 8fg6 with every lever off), confidence head rerun under lever ablations (hook/sitecustomize.py)
# mem: 60
# threads: 4
# cap: 7200
R=/home/ubuntu/spd/mol1-whl3; I=/home/ubuntu/spd/mol1-quality/inputs; T=/home/ubuntu/spd/mol1-wh1024cap/tree-59893ab61
export TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
rc=0
cell() { # arm case seed [env]: arm main = the served lever set, nolev = TT_BIO_LEVERS=none (every lever off)
  d=$R/out/$1/.runs/$2/seed_$3; grep -qs rc=0 $d/rc && return 0; mkdir -p $d; rm -f $R/out/dump/$1_$2_s$3/arms.part.pt
  TREE=$T bash $R/run_cell.sh $CHIP $d 5400 -- bash -c "export PYTHONPATH=$R/hook:\$PYTHONPATH WHL3_DUMP=$R/out/dump/$1_$2_s$3 $4; cd $I && python -c \"from tt_bio.main import cli; cli()\" predict $2.json --model mol1 --seed $3 --debug --out_dir $R/out/$1"
  r=$?; echo rc=$r > $d/rc; [ $r = 0 ] || rc=$r
}
cell main 8fg6-assembly1 42
cell nolev 8fg6-assembly1 42 TT_BIO_LEVERS=none
cell main 8k83-assembly1 42
exit $rc
