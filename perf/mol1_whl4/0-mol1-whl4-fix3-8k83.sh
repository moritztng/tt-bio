# mol1-whl4: 8k83 s42 on 8c8b2ea72 (trunk single track fp32 end to end, fp32 layer norm statistics from elementwise adds)
# mem: 60
# threads: 4
# cap: 2400
R=/home/ubuntu/spd/mol1-whl4
export PATH=/home/ubuntu/spd/mol1-quality/kalign/bin:$PATH TREE=$R/tree-8c8b2ea72 TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
d=$R/out/fix-8c8b2ea72/8k83; mkdir -p $d
bash $R/run_cell.sh $CHIP $d 2400 -- env PYTHONPATH=/home/ubuntu/spd/mol1-whl3/hook:$TREE WHL3_DUMP=$d/arms WHL3_ARMS=served WHL3_LAYERS=1 bash -c "cd /home/ubuntu/spd/mol1-quality/inputs && python -c \"from tt_bio.main import cli; cli()\" predict 8k83-assembly1.json --model mol1 --seed 42 --out_dir $d/fold"
r=$?; echo rc=$r > $d/rc; exit $r
