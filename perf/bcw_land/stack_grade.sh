#!/bin/bash
# stack_grade.py on base (.base/, 8c3783b87) then this tree, one card. $1 = card.
here=$(cd "$(dirname "$0")/../.." && pwd); card=$1; o=$here/perf/bcw_land/out/grade; mkdir -p $o
for arm in ${ARMS:-base stack}; do
    tree=$here; [ $arm = base ] && tree=$here/.base
    echo "=== $arm $(git -C $tree rev-parse --short HEAD) $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"
    (cd $tree && TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-land \
        PYTHONPATH=$tree:$HOME/bcx_e2e/bc2 timeout 1500 ~/bcx_e2e_venv/bin/python3 -u $here/perf/bcw_land/stack_grade.py bcw_land_$arm) > $o/$arm.txt 2>&1
    echo "  rc=$? $(date -u +%FT%TZ)"
done
