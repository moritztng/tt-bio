#!/usr/bin/env bash
# One-shot finalize: reduce, report, negative controls, assemble the README, write the state doc.
set -uo pipefail
cd /home/ttuser/.coworker/wt/c10-fixed-cost
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH PYTHONPATH=$PWD
unset TT_METAL_HOME TT_METAL_RUNTIME_ROOT LD_LIBRARY_PATH
RUN=${1:-sweep1}
D=perf/c10_fixed_cost/runs/$RUN
mkdir -p perf/c10_fixed_cost/tmp
echo "===== REDUCE ====="
python3 perf/c10_fixed_cost/reduce.py "$D" --out "$D/analysis.json"; echo "reduce rc=$?"
echo "===== REPORT ====="
python3 perf/c10_fixed_cost/report.py "$D/analysis.json" | tee "$D/report.md"
echo "===== NEGATIVE CONTROLS ====="
for S in 512 298; do python3 perf/c10_fixed_cost/negative_control.py "$D" $S; done \
  2>&1 | tee perf/c10_fixed_cost/negative_control.log | tail -25
echo "===== README ====="
{ cat perf/c10_fixed_cost/README_head.md; echo; echo "## Measured"; echo;
  echo "One benchlocked session, run \`$RUN\`, qb2 card 0. Cells are accepted warm folds only.";
  echo; cat "$D/report.md"; } > perf/c10_fixed_cost/README.md
wc -l perf/c10_fixed_cost/README.md
echo "===== STATE DOC ====="
python3 perf/c10_fixed_cost/state_doc.py "$D/analysis.json" "$D" \
  /home/ttuser/.coworker/state/c10-fixed-cost.md > /dev/null && echo "state doc written"
wc -c /home/ttuser/.coworker/state/c10-fixed-cost.md
echo "===== SIZES ====="
du -sh "$D" "$D"/512 "$D"/298 2>/dev/null
du -ch "$D"/*/clock.jsonl.gz "$D"/*/holders.jsonl.gz "$D"/ambient.jsonl.gz 2>/dev/null | tail -1
du -ch "$D"/*/cifs 2>/dev/null | tail -1
