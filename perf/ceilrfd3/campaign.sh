#!/bin/bash
# The whole RFD3 ceiling campaign in the order that answers the most per chip claimed.
#
# Bit-exactness first, because a ceiling reached by a path that is wrong at 256 residues is
# a regression, and 256 is what users run. Then the OLD ceiling with its negative control on
# the unblocked path, so "490" is this box's measurement and not an inherited number. Then
# the new ladder.
set -u
cd "$(dirname "$0")/../.." || exit 1
export SRC=$PWD
END=${END:-$(( $(date +%s) + 21600 ))}

CMD=perf/ceilrfd3/rfd3_init_digest.py TAG=digest_fix DEADLINE=$END STOP_ON_FAIL=0 \
  bash perf/ceilrfd3/chain.sh 128 256 480
CMD=perf/ceilrfd3/rfd3_init_digest.py TAG=digest_base DEADLINE=$END STOP_ON_FAIL=0 \
  EXTRA_ENV=TT_BIO_ATOM_PAIR_BUDGET_BYTES=0 bash perf/ceilrfd3/chain.sh 128 256 480

TAG=base DEADLINE=$END EXTRA_ENV=TT_BIO_ATOM_PAIR_BUDGET_BYTES=0 \
  bash perf/ceilrfd3/chain.sh 448 480 512 544

TAG=fix DEADLINE=$END bash perf/ceilrfd3/chain.sh 512 576 640 704 768 832 896 960 1024
echo "[campaign] $(date -Is) done" >> "$SRC/perf/ceilrfd3/results/campaign.log"
