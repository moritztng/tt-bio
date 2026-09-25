cd /home/ttuser/.coworker/wt/bcx-tracewire
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=1,2 TT_BIO_LEASE_HOLDER=worker:bcx-tracewire
PY=/home/ttuser/bcx_e2e_venv/bin/python3
for arm in trace eager; do
  timeout 1500 $PY perf/bcx_tracewire/round_ab.py --arm $arm --digest --rounds 7 --seed 100 --card 2 \
    --project perf/bcx_tracewire/runs/digest_${arm}_seed100 --out round_digest_${arm}_seed100.json \
    > perf/bcx_tracewire/logs/digest_${arm}_s100.log 2>&1 < /dev/null
  echo "$arm exit $?" >> perf/bcx_tracewire/logs/digest_pair.status
done
