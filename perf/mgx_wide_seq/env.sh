# source perf/mgx_wide_seq/env.sh <card>  -- the pinned whglx environment for this row
card=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$root
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
py=$HOME/env/bin/python
