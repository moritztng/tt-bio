#!/bin/sh
# Device runner for whglx (Wormhole Galaxy, j10glx02). Card 0 is this row's grant; the open is
# pinned because an unpinned open on this box brings every chip up and breaks its co-tenants.
cd /home/agent/wt-bhenv
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:roof-bh-envelopes-on-wh
export TT_BIO_LEASE_DIR=/home/agent/leases TT_BIO_LEASE_TIMEOUT=900
export PYTHONPATH=/home/agent/wt-bhenv/tt-bio
exec /home/agent/env/bin/python3 "$@"
