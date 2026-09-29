cd /home/ttuser/.coworker/wt/bc2-teardown-next-job
export PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2 TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bc2-teardown-next-job
PY=/home/ttuser/bcx_e2e_venv/bin/python3
echo "commit=$(git rev-parse --short HEAD) start=$(date -u +%FT%TZ)"
timeout 1500 $PY -m pytest tests/test_bindcraft2.py -q -p no:cacheprovider > perf/bc2_teardown/out/land/host.log 2>&1; echo host_rc=$?
timeout 1800 $PY -m pytest tests/test_bindcraft2_hw.py -q -s -p no:cacheprovider > perf/bc2_teardown/out/land/hw.log 2>&1; echo hw_rc=$?
echo end=$(date -u +%FT%TZ)
