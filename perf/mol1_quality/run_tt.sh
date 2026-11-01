#!/usr/bin/env bash
# tt-bio Mol-1 on the frozen quality inputs, upstream defaults (20 cycles, 200 steps, 5 samples), one process per
# leased chip, cases sharded round-robin. Writes upstream's layout, which grade.py reads.
#   bash run_tt.sh [--reverse] [--book OWNER] [--cases CASES.json] INPUTS OUT "42 66 101 2024 8888" CHIP [CHIP ...]
# Chip workers take case+seed folds by claim file, so several invocations on one host can share OUT. --reverse walks
# the case list from the end (a second host starts there; its finished .runs are copied back and skipped). --book
# OWNER: a chip folds only while the box's arbiter (~/spd/feed/book.py) gives its card to OWNER, and stops when not.
# --cases: only the case ids a CASES.json lists under "paired" (an A/B subset).
# INPUTS = quality_inputs.py tree (cwd for every run, so MSA/FILE_ paths resolve). Restartable: a case+seed with
# rc=0 is skipped. AICLK of every chip at 1 Hz goes to OUT/aiclk.log for the whole run (accuracy does not depend
# on it; the record carries it anyway).
set -uo pipefail
REV=0; OWNER=; SUB=
while [[ ${1:-} == --* ]]; do case $1 in --reverse) REV=1; shift ;; --book) OWNER=$2; shift 2 ;; --cases) SUB=$(realpath "$2"); shift 2 ;; *) echo "bad flag $1"; exit 2 ;; esac; done
IN=$(cd "${1:?inputs}" && pwd); OUT=$(mkdir -p "${2:?out}" && cd "$2" && pwd); SEEDS=${3:?seeds}; shift 3
HOLDER=${TT_BIO_LEASE_HOLDER:-worker:mol1-quality}; LOCK_HOST=${LOCK_HOST:-qb2}
CHIPS=("$@"); [ ${#CHIPS[@]} -gt 0 ] || { echo "no chip"; exit 2; }
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!*/tt_aiclk 2>/dev/null | tr '\n' ' ')"; sleep 1; done ) \
  >> "$OUT/aiclk.log" & CLK=$!; trap 'kill $CLK 2>/dev/null' EXIT
# case file -> recipe flags, from CASES.json via INDEX.json (default | use_msa_false | use_template_true | use_rna_msa_true)
mapfile -t JOBS < <(python3 - "$IN" "$SUB" <<'PY'
import json, sys
from pathlib import Path
idx = json.loads((Path(sys.argv[1]) / "INDEX.json").read_text())
rec = idx.get("recipes", {})
sub = {c["id"] for c in json.loads(Path(sys.argv[2]).read_text())["paired"]} if sys.argv[2] else None
for cid, f in sorted(idx["cases"].items()):
    if cid in idx.get("headline_only", ()) or (sub is not None and cid not in sub):  # GPU-only set / not in the subset
        continue
    print(f, {"use_msa_false": "--single_sequence", "use_template_true": "--use_template",
              "use_rna_msa_true": "--use_rna_msa"}.get(rec.get(cid, "default"), ""))
PY
)
[ $REV = 1 ] && mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | tac)
worker() {  # chip
  local chip=$1 me=$BASHPID
  for job in "${JOBS[@]}"; do
    read -r f flags <<< "$job"; name=${f%.json}
    for s in $SEEDS; do
      d=$OUT/.runs/$name/seed_$s; [ -f "$d/rc" ] && grep -q "rc=0" "$d/rc" && continue
      if [ -n "$OWNER" ] && [ "$(python3 ~/spd/feed/book.py active "$chip" 2>/dev/null)" != "$OWNER" ]; then
        echo "$(date -u +%FT%TZ) TT-STOP chip=$chip window over"; return 0
      fi
      mkdir -p "$d"
      if ! ( set -o noclobber; echo "$me" > "$d/claim" ) 2>/dev/null; then  # held, unless its owner died
        kill -0 "$(cat "$d/claim" 2>/dev/null)" 2>/dev/null && continue
        echo "$me" > "$d/claim"
      fi
      ( cd "$IN" && TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip TT_BIO_LEASE_HOLDER=$HOLDER \
          timeout 5400 tt-bio predict "$f" --model mol1 --seed "$s" \
          --out_dir "$OUT" $flags ) > "$d/log" 2>&1
      echo "rc=$?" > "$d/rc"; rm -f "$d/claim"; echo "$(date -u +%FT%TZ) TT-DONE chip=$chip $name $s $(cat "$d/rc")"
    done
  done
}
# each chip worker holds the fleet card flock (~/spd_<host>_card<N>.lock) for its whole run
W=()
for i in "${!CHIPS[@]}"; do
  ( exec 9> "$HOME/spd_${LOCK_HOST}_card${CHIPS[$i]}.lock"
    flock -n 9 || { echo "chip ${CHIPS[$i]}: card lock held"; exit 3; }
    worker "${CHIPS[$i]}" ) & W+=($!)
done
wait "${W[@]}"  # not the AICLK sampler
echo "TT-ALL-DONE"
