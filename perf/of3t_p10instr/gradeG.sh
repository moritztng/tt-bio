#!/bin/bash
# The amended grade (AMENDMENT-one-process-per-checkpoint.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/out"
arms=(TF7 B2trunk I_TFs1 I_D12 X12b I_D12s1)
side() { local s=$1 out="F_L40G_$1.json+F_L40Gs1_$1.json"; for a in "${arms[@]}"; do out+="+G_${a}_$s.json"; done; echo "$out"; }
python3 ../grade.py ab --history "$(side a),$(side b)" \
    --design two=G_TF7_a.json+G_B2trunk_a.json+G_I_TFs1_a.json:TF7:B2trunk:I_TFs1 \
    --design twelve=G_I_D12_a.json+G_X12b_a.json+G_I_D12s1_a.json:I_D12:X12b:I_D12s1 \
    --out VERDICT_G.json
