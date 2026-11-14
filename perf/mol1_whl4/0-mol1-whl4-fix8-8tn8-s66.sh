# mol1-whl4 gate refold: 8tn8 s66 on bfb187307 (fp32 trunk single track + host fp32 confidence single track)
# mem: 60
# threads: 4
# cap: 2400
R=/home/ubuntu/spd/mol1-whl4
export PATH=/home/ubuntu/spd/mol1-quality/kalign/bin:$PATH TREE=$R/tree-bfb187307 TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
d=$R/out/fix8-bfb187307/8tn8-s66; mkdir -p $d
bash $R/run_cell.sh $CHIP $d 2400 -- bash -c "cd /home/ubuntu/spd/mol1-quality/inputs && python -c \"from tt_bio.main import cli; cli()\" predict 8tn8-assembly1.json --model mol1 --seed 66 --out_dir $d/fold"
r=$?; echo rc=$r > $d/rc; exit $r
