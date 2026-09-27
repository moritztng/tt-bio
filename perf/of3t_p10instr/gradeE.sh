#!/bin/bash
# The 256-draw two-step grade (PREREG-256.md). twelve stays graded by gradeD.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/out"
arms=(TF7 B2trunk I_TFs1 I_D12 X12b I_D12s1)
two=(TF7 B2trunk I_TFs1)
side() { local p=$1 s=$2; shift 2; local out=""; for a in "$@"; do out+="${p}_${a}_$s.json+"; done; echo "${out%+}"; }
python3 ../grade.py ab --draws 256 \
    --history "$(side D256 a "${two[@]}"),$(side D128 a "${arms[@]}"),F_L40G_a.json+F_L40Gs1_a.json+$(side G a "${arms[@]}"),F_L40G_b.json+F_L40Gs1_b.json+$(side G b "${arms[@]}")" \
    --design two=D256_TF7_a.json+D256_B2trunk_a.json+D256_I_TFs1_a.json:TF7:B2trunk:I_TFs1 \
    --bar two=0.019428875906963877 \
    --out VERDICT_D256.json
