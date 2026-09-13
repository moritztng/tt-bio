#!/usr/bin/env bash
# base, l1, base -- the A/A control is inside the run, not a separate campaign.
# Two base arms that disagree with each other make the base-vs-l1 compare unreadable, and
# BoltzGen has no seed of its own, so the control has to be measured here.
set -euo pipefail
WT="${WT:-$HOME/wt-bgatoml1}"
# whglx runs two unix accounts against one Galaxy and /tmp/tt-bio-device-leases belongs to
# whichever created it. It is tt-admin's today, mode 664, so this account cannot take a lease
# there at all -- TT_BIO_LEASE_DIR points at the one this account can write.
CARD="${CARD:-1}"
SPEC="${SPEC:-$WT/tests/fixtures/boltzgen/bg400.yaml}"
ROOT="${ROOT:-$HOME/scratch/bgatoml1}"
SEED="${SEED:-0}"
mkdir -p "$ROOT"

# The spec names its target by a relative path, so it has to run from a directory that has both.
WORK="$ROOT/spec"
mkdir -p "$WORK"
cp "$SPEC" "$WORK/"
cp "$(dirname "$SPEC")/$(grep -o '[A-Za-z0-9_]*\.cif' "$SPEC" | head -1)" "$WORK/"

# ARMS picks which arms run. The A/A control (base2) is what makes a digest compare readable, so
# it is the default; a rung being bisected only needs "base l1", because the A/A floor is a
# property of the seeding and was measured exact at bg400.
for arm in ${ARMS:-base l1 base2}; do
  a="${arm%2}"
  out="$ROOT/$arm"
  echo "=== $(date -Is) arm=$arm ==="
  rm -rf "$out"
  set +e
  TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER=worker:b2z2-boltzgen-atoml1-leg \
  TT_BIO_LEASE_DIR="${TT_BIO_LEASE_DIR:-$HOME/leases}" \
  PYTHONPATH="$WT" \
  env -u TT_METAL_DEVICE_PROFILER \
  "$HOME/env/bin/python" "$WT/perf/b2z2_bgatoml1/bg_arm.py" \
      --spec "$WORK/$(basename "$SPEC")" --out "$out" --arm "$a" \
      --device "$CARD" --seed "$SEED" 2>&1 | tee "$ROOT/$arm.log"
  set -e
done

case " ${ARMS:-base l1 base2} " in
  *" base2 "*) "$HOME/env/bin/python" "$WT/perf/b2z2_bgatoml1/compare.py" "$ROOT" ;;
  *) "$HOME/env/bin/python" "$WT/perf/b2z2_bgatoml1/compare.py" "$ROOT" --arms base l1 ;;
esac
