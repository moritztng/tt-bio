#!/bin/bash
# The 128-draw grade (PREREG-128.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/out"
arms=(TF7 B2trunk I_TFs1 I_D12 X12b I_D12s1)
side() { local p=$1 s=$2 out=""; for a in "${arms[@]}"; do out+="${p}_${a}_$s.json+"; done; echo "${out%+}"; }
python3 ../grade.py ab --draws 128 \
    --history "$(side D128 a),F_L40G_a.json+F_L40Gs1_a.json+$(side G a),F_L40G_b.json+F_L40Gs1_b.json+$(side G b)" \
    --design two=D128_TF7_a.json+D128_B2trunk_a.json+D128_I_TFs1_a.json:TF7:B2trunk:I_TFs1 \
    --design twelve=D128_I_D12_a.json+D128_X12b_a.json+D128_I_D12s1_a.json:I_D12:X12b:I_D12s1 \
    --bar two=0.019428875906963877 --bar twelve=0.04507007146777562 \
    --out VERDICT_D128.json
