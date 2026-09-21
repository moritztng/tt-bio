#!/usr/bin/env bash
# AICLK on card 0 sampled DURING a window, from qbcard/cardtel.tsv (2 s cadence).
# usage: clockwin.sh <start_epoch> <end_epoch>
set -euo pipefail
S=$1; E=$2
awk -F'\t' -v s="$S" -v e="$E" '
  $1 ~ /^[0-9]/ && $1+0 >= s && $1+0 <= e { n++; c=$9+0; t+=c; if (mn=="" || c<mn) mn=c; if (c>mx) mx=c }
  END { if (n==0) { print "NO SAMPLES IN WINDOW"; exit 1 }
        printf "card0 AICLK during the window: n=%d mean=%.0f MHz min=%d max=%d\n", n, t/n, mn, mx }
' /home/ttuser/qbcard/cardtel.tsv
