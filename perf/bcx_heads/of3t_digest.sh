#!/usr/bin/env bash
# OF3T's own whole-backward gradient digest, run from wk/of3t-cropwall with the unified verbs
# transplanted (post) and from of3t-cropwall's own fix via --taped-from (pre), same process
# protocol, card 3 on qb1.
set -u
WT=/home/ttuser/.coworker/wt/bcx-heads
cd "$WT/.of3t/cw"
run() {
  echo "=== $1 start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-heads \
    timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_cropwall/grad_digest.py "${@:2}" 2>&1 \
    | grep -v "DEBUG    \|Initial ttnn.CONFIG\|^Config{" | tail -25
  echo "=== $1 end $(date -u +%FT%TZ) rc=${PIPESTATUS[0]}"
}
for arm in "$@"; do
  case $arm in
    post) run post --tokens 256 --out "$WT/perf/bcx_heads/of3t_grads_256_unified.json" ;;
    pre)  run pre  --tokens 256 --taped-from e547b321d --out "$WT/perf/bcx_heads/of3t_grads_256_cropwall.json" ;;
    base) run base --tokens 256 --taped-from 9de66106e --out "$WT/perf/bcx_heads/of3t_grads_256_base.json" ;;
  esac
done
