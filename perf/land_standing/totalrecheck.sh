#!/bin/bash
# Re-verify every lever this row's TOTAL counts, against CURRENT origin/main.
#
# The TOTAL is this row's headline number and it rests on four levers still being present and
# default-ON. main has moved many commits since the audit was written, and a lever that was
# default-on when audited is not necessarily default-on now -- that is exactly what
# `merged-lever-defaults-off-is-not-a-landed-win` is about, read forwards.
#
# Checks the SYMBOL on main, never a remembered line number: line numbers move, and this row has
# already been bitten by a comment whose reason had drifted away from its flag.
set -u
cd /home/ttuser/.coworker/wt/land-standing || exit 1
M=/tmp/main_recheck
rm -rf $M && mkdir -p $M && git archive origin/main tt_bio | tar -x -C $M

echo "origin/main = $(git rev-parse --short origin/main)"
echo

chk () {
  label=$1
  pat=$2
  file=$3
  hit=$(grep -n "$pat" "$M/tt_bio/$file" 2>/dev/null | head -1)
  if [ -n "$hit" ]; then
    echo "  OK    $label"
    echo "        $file:$(echo "$hit" | cut -c1-110)"
  else
    echo "  MISS  $label  -- pattern '$pat' not found in $file"
  fi
}

echo "1.438 s  derived fused _MM_BLOCK key"
chk "self-pair fix present (widths[i + 1:])" 'widths\[i + 1:\]' tenstorrent.py
n=$(grep -c 'widths\[i:\]' $M/tt_bio/tenstorrent.py 2>/dev/null || echo 0)
echo "        regression check: widths[i:] occurrences = $n (must be 0)"
echo
echo "11.564 s  fused HiFi triangle attention at openfold3.trunk"
chk "site default True" 'triatt_sdpa_hifi_site("openfold3.trunk", True)' openfold3_trunk.py
echo
echo "0.1315 s  K4 TT_BIO_SDPA_BAND_DIV_K   (Blackhole-only: gated on not _IS_SMALL_GRID)"
chk "env_flag default True" 'env_flag("TT_BIO_SDPA_BAND_DIV_K", True)' tenstorrent.py
chk "grid gate still present" '_SDPA_BAND_DIV_K and not _IS_SMALL_GRID' tenstorrent.py
echo
echo "9.5000 s  TT_BIO_TRIATT_NARROW_Q_FALLBACK"
chk "env_flag default True" 'env_flag("TT_BIO_TRIATT_NARROW_Q_FALLBACK", True)' tenstorrent.py
