#!/usr/bin/env bash
# tt-bio Mol-1 on the frozen quality inputs, upstream defaults (20 cycles, 200 steps, 5 samples), one process per
# leased chip, cases sharded round-robin. Writes upstream's layout, which grade.py reads.
#   bash run_tt.sh INPUTS OUT "42 66 101 2024 8888" CHIP [CHIP ...]
# INPUTS = quality_inputs.py tree (cwd for every run, so MSA/FILE_ paths resolve). Restartable: a case+seed with
# rc=0 is skipped. AICLK of every chip at 1 Hz goes to OUT/aiclk.log for the whole run (accuracy does not depend
# on it; the record carries it anyway).
set -uo pipefail
IN=$(cd "${1:?inputs}" && pwd); OUT=$(mkdir -p "${2:?out}" && cd "$2" && pwd); SEEDS=${3:?seeds}; shift 3
CHIPS=("$@"); [ ${#CHIPS[@]} -gt 0 ] || { echo "no chip"; exit 2; }
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!*/tt_aiclk 2>/dev/null | tr '\n' ' ')"; sleep 1; done ) \
  >> "$OUT/aiclk.log" & CLK=$!; trap 'kill $CLK 2>/dev/null' EXIT
# case file -> recipe flags, from CASES.json via INDEX.json (default | use_msa_false | use_template_true | use_rna_msa_true)
mapfile -t JOBS < <(python3 - "$IN" <<'PY'
import json, sys
from pathlib import Path
idx = json.loads((Path(sys.argv[1]) / "INDEX.json").read_text())
rec = idx.get("recipes", {})
for cid, f in sorted(idx["cases"].items()):
    if cid in idx.get("headline_only", ()):  # GPU-only set
        continue
    print(f, {"use_msa_false": "--use_msa false", "use_template_true": "--use_template true",
              "use_rna_msa_true": "--use_rna_msa true"}.get(rec.get(cid, "default"), ""))
PY
)
worker() {  # chip index-in-list
  local chip=$1 i=$2 n=${#CHIPS[@]} k=0
  for job in "${JOBS[@]}"; do
    k=$((k + 1)); [ $(((k - 1) % n)) -eq "$i" ] || continue
    read -r f flags <<< "$job"; name=${f%.json}
    for s in $SEEDS; do
      d=$OUT/.runs/$name/seed_$s; [ -f "$d/rc" ] && grep -q "rc=0" "$d/rc" && continue
      mkdir -p "$d"
      ( cd "$IN" && TT_VISIBLE_DEVICES=$chip timeout 5400 tt-bio predict "$f" --model mol1 --seed "$s" \
          --out_dir "$OUT" $flags ) > "$d/log" 2>&1
      echo "rc=$?" > "$d/rc"; echo "$(date -u +%FT%TZ) TT-DONE chip=$chip $name $s $(cat "$d/rc")"
    done
  done
}
for i in "${!CHIPS[@]}"; do worker "${CHIPS[$i]}" "$i" & done
wait
echo "TT-ALL-DONE"
