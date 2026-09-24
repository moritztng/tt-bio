#!/bin/bash
# After chain_infab.sh exits: price the protenix clamps and the openfold3 head uploads on card 1.
W=/home/ttuser/.coworker/wt/of3t-infab; P=$W/perf/of3t_infab; S=/tmp/of3t/of3t-infab; CARD=1
log() { echo "=== $* $(date -u +%FT%TZ)" >> $P/chain.log; }
while kill -0 $1 2>/dev/null; do sleep 30; done
[ -s $P/PRICE.json ] && exit 0
if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q "/dev/tenstorrent/$CARD\$"; then log "price: card $CARD held, not run"; exit 1; fi
( cd $S/after && env -u TT_MESH_GRAPH_DESC_PATH TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_LEASE_HOLDER=worker:of3t-infab PYTHONPATH=$S/after OMP_NUM_THREADS=8 \
    timeout 1800 /home/ttuser/tt-bio-dev/env/bin/python3 $P/price.py $P/PRICE.json $CARD > $S/price.log 2>&1 )
log "price exit $? $(tail -1 $S/price.log)"
