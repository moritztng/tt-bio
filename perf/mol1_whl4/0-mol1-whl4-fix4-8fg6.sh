# mol1-whl4: 8fg6 s42 on 6f84a49fa (trunk + confidence head single track fp32, LN statistics from fp32 adds)
# mem: 60
# threads: 4
# cap: 2400
R=/home/ubuntu/spd/mol1-whl4
export PATH=/home/ubuntu/spd/mol1-quality/kalign/bin:$PATH TREE=$R/tree-6f84a49fa TT_BIO_CACHE=/home/ubuntu/spd/mol1-port/cache BOLTZ_CACHE=/home/ubuntu/spd/mol1-port/cache SHARE=4
d=$R/out/fix-6f84a49fa/8fg6; mkdir -p $d
bash $R/run_cell.sh $CHIP $d 2400 -- env PYTHONPATH=/home/ubuntu/spd/mol1-whl3/hook:$TREE WHL3_DUMP=$d/arms WHL3_ARMS=served WHL3_LAYERS=1 bash -c "cd /home/ubuntu/spd/mol1-quality/inputs && python -c \"from tt_bio.main import cli; cli()\" predict 8fg6-assembly1.json --model mol1 --seed 42 --out_dir $d/fold"
r=$?; echo rc=$r > $d/rc; exit $r
