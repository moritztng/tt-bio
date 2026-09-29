#!/bin/bash
# Run trials in order, one at a time:  chain.sh card:kind:tag ...
cd "$(dirname "$0")"
for t in "$@"; do IFS=: read -r c k g <<< "$t"; ./trial.sh "$c" "$k" "$g"; done
echo "chain done $(date -u +%FT%TZ)"
