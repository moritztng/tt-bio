#!/bin/bash
# Harvest every 1536 scRMSD this row is waiting on, from wherever it lives, in one command.
#
# Three arms answer three different questions and they are easy to conflate, so this script
# labels each one with the SIDE it belongs to and the report keys on that:
#
#   device        the card designed it and the card refolded it   (whglx, both targets)
#   upstream-refold-of-device-design
#                 the card designed it, upstream fp32 refolded it (qb2, gpb target)
#   upstream-cpu-fp32-sixstep
#                 upstream designed AND refolded it               (qb2, big_1831 target)
#
# The middle one is the attribution split: if the device's own number is worse than 512 but
# the upstream refold of the SAME designs agrees with it, the designs are worse and the fold
# that scores them is fine. If they disagree, it is the fold. Nothing else available without
# a GPU separates those.
#
# Everything is read at step 4 (`refold_design_cif/`), which `scrmsd.py --check` reproduces
# against the pipeline's own column to 0.00000 A on the device and 0.00001 A on upstream.
#
#   bash perf/mgxaccuracy/collect_1536.sh [OUT_JSONL]
set -u
OUT=${1:-perf/mgxaccuracy/results/size_1536.jsonl}
GPB=perf/mgxaccuracy/targets/gpb_dimer_1646.cif
: > "$OUT"

say() { printf '\n== %s\n' "$1"; }

say "device arms on whglx (step 4 = refold_design_cif)"
for t in gpb q; do
    case $t in gpb) tgt=$GPB;; q) tgt="big_1831 crop, offset 0";; esac
    n=$(ssh -o BatchMode=yes whglx \
        "ls ~/mgxacc-work/out_boltzgen_1536_d8_s400_$t/intermediate_designs_inverse_folded/refold_design_cif/*.cif 2>/dev/null | wc -l")
    echo "  $t: $n of 8 refolded"
    [ "${n:-0}" -gt 0 ] || continue
    ssh -o BatchMode=yes whglx "cd ~/wt-mgx-design-accuracy && \$HOME/env/bin/python \
        perf/mgxaccuracy/scrmsd.py ~/mgxacc-work/out_boltzgen_1536_d8_s400_$t --json \
        --model boltzgen --target '$tgt' --target-res 1536 --side device" >> "$OUT"
done

say "upstream fp32 refolding the DEVICE's 1536 designs (qb2)"
n=$(ssh -o BatchMode=yes qb2 \
    "ls ~/bgref-work/out_dev1536/intermediate_designs_inverse_folded/refold_design_cif/*.cif 2>/dev/null | wc -l")
echo "  $n of 8 refolded"
if [ "${n:-0}" -gt 0 ]; then
    ssh -o BatchMode=yes qb2 "cd ~/mgxacc-tools && ~/bgref-env/bin/python3 scrmsd.py \
        ~/bgref-work/out_dev1536 --json --model boltzgen --target '$GPB' --target-res 1536 \
        --side upstream-refold-of-device-design" >> "$OUT"
fi

say "upstream's own 1536 design, if its design step has finished (qb2)"
n=$(ssh -o BatchMode=yes qb2 \
    "ls ~/bgref-work/out1536_df/intermediate_designs_inverse_folded/refold_design_cif/*.cif 2>/dev/null | wc -l")
echo "  $n refolded (run ref/designfolding_only.sh --size 1536 first if 0)"
if [ "${n:-0}" -gt 0 ]; then
    ssh -o BatchMode=yes qb2 "cd ~/mgxacc-tools && ~/bgref-env/bin/python3 scrmsd.py \
        ~/bgref-work/out1536_df --json --model boltzgen --target 'big_1831 crop, offset 0' \
        --target-res 1536 --side upstream-cpu-fp32-sixstep" >> "$OUT"
fi

say "report — across-target band FIRST, then the size comparison"
python3 perf/mgxaccuracy/report.py perf/mgxaccuracy/results/*.jsonl --model boltzgen
echo
echo "Read in this order: the 512 across-target floor is 2.10 A (medians 8.46 / 10.56), so a"
echo "1536 median inside it says nothing about size. n=8 per size gives 45 % power against a"
echo "doubling (results/power.txt), so 'inside' means this study could not see an effect."
