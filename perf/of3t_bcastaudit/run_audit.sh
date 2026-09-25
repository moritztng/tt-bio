#!/usr/bin/env bash
# One counted fold: run_audit.sh <name> <tt-bio or python argv...>
# Counts land in $ROOT/<name>/counts/<pid>.json (see sitecustomize.py); the fold's own output in $ROOT/<name>/out.
set -euo pipefail
WT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${ROOT:-/home/ttuser/of3t_bcastaudit}
name=$1; shift
rm -rf "$ROOT/$name"; mkdir -p "$ROOT/$name/counts"
export BCASTAUDIT_DIR=$ROOT/$name/counts PYTHONPATH=$WT/perf/of3t_bcastaudit:$WT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-bcastaudit
cd "$WT"
"$@" > "$ROOT/$name/log.txt" 2>&1
