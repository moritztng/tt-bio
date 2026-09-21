#!/usr/bin/env bash
# AICLK on any card sampled DURING a window, from qbcard/cardtel.tsv (2 s cadence).
# usage: clockwin.sh <card> <start_epoch> <end_epoch>
#
# of3t_residual/clockwin.sh hardcodes card 0's column. This row runs on card 3, so the column is
# resolved from the header by NAME -- a hardcoded index would read a different card's clock and
# still print a plausible number, which is the one failure a clock discipline cannot survive.
set -euo pipefail
C=$1; S=$2; E=$3
T=/home/ttuser/qbcard/cardtel.tsv
awk -F'\t' -v card="c${C}_tt_aiclk" -v s="$S" -v e="$E" '
  NR==1 { for (i=1; i<=NF; i++) if ($i == card) col=i
          if (!col) { print "NO COLUMN " card " IN HEADER"; exit 1 } ; next }
  $1 ~ /^[0-9]/ && $1+0 >= s && $1+0 <= e { n++; c=$col+0; t+=c; if (mn=="" || c<mn) mn=c; if (c>mx) mx=c }
  END { if (n==0) { print "NO SAMPLES IN WINDOW"; exit 1 }
        printf "card%s AICLK during the window: n=%d mean=%.0f MHz min=%d max=%d\n", substr(card,2,1), n, t/n, mn, mx }
' "$T"
