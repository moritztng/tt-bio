#!/bin/bash
# Everything that turns a finished campaign into the deliverable's numbers, in one command and in the
# right order. Run it after e2e/campaign.done exists.
#
#   bash /home/ttuser/relion-scratch/e2e_harvest.sh
#
# It refuses to run on an arm that did not actually run. benchlock exits 75 when it times out waiting
# for the box and deliberately does NOT measure, and an arm that exited that way leaves a log and no
# useful output -- harvesting it would produce a table of nothing that looks like a table.
set -u
S=/home/ttuser/relion-scratch
cd "$S" || exit 1

ok=1
for arm in ref tt; do
  if [ ! -f "$S/e2e/$arm.rc" ]; then echo "$arm: no rc, the arm has not finished"; ok=0; continue; fi
  rc=$(cat "$S/e2e/$arm.rc")
  echo "$arm: $rc"
  case "$rc" in
    rc=0*) ;;
    *) echo "  ^ not a clean run; fix before harvesting"; ok=0 ;;
  esac
done
[ "$ok" = 1 ] || { echo "REFUSING to harvest"; exit 1; }

echo
echo "################ 1. by-stage wall-clock split, both arms"
python3 "$S/e2e_stages.py" "$S/e2e/ref.log" "$S/e2e/tt.log"

echo
echo "################ 2. is it the same answer? FSC, cross-FSC, assignments, compounding"
python3 "$S/e2e_compare.py"

echo
echo "################ 3. the number RELION itself prints"
bash "$S/e2e_postprocess.sh"

echo
echo "################ 4. the walls, for the A/B"
echo "ref (RELION's own kernels): $(cat "$S/e2e/ref.time")"
echo "tt  (coarse through the bridge): $(cat "$S/e2e/tt.time")"
echo "fields are: wall user sys maxrss_kB"
