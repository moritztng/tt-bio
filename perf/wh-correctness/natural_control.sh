#!/bin/bash
# A natural protein at the top of OpenDDE's served range, against Protenix-v2 on the same
# input, card and session.
#
# Why this exists: every rung of the tear ladder used `matrix.cdk2(N)`, the 298 aa CDK2 domain
# tiled and truncated. Above 384 that is a tandem duplicate, and a tandem duplicate is a
# genuinely ambiguous packing problem — so a clash fraction measured on it cannot separate
# "the model is wrong" from "the input is adversarial". The discriminator is a real protein at
# the same length with a healthy model as the control, and the number to read is the RATIO to
# that control, not the absolute fraction: `check_structure.py`'s 0.1 % clash bar is stricter
# than any of these models achieves, and the control itself moves with the input (0.24 % on
# tiled 512, 0.49 % on natural 531).
#
# Result, 2026-08-17, tt-bio 1ea1e6f3b, pc card 0, single sequence, recycle 10 / 200 steps:
#   PKM P14618, 531 aa   opendde 43 clashes (1.06 %), rg 1.034  |  protenix-v2 20 (0.49 %), rg 0.975
#   luciferase P08659, 550 aa  opendde 135 (3.15 %), rg 0.930   |  protenix-v2 HUNG at trunk 9/10, twice
# 2.2x the control at 531 aa is the same band opendde shows at 128-384 aa (1.2-2.5x), against
# 6-9x on tiled 512/544. 550 aa is above the published 544 cap and has no control.
#
# Sequences come from UniProt at run time on purpose: a hand-pasted sequence in a script is a
# fabrication risk, and these are the reference entries.
set -u
WT=${WT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/home/moritz/tt-bio/env/bin/python3}
CARD=${TT_VISIBLE_DEVICES:-0}
HOLDER=${TT_BIO_LEASE_HOLDER:-worker:natural-control}
OUT=${OUT:-/tmp/natural_control}
# UniProt accession : length : stem.  531 is inside OpenDDE's 544 cap; 550 is just above it.
TARGETS=${TARGETS:-"P14618:531:pkm_531 P08659:550:luci_550"}
MODELS=${MODELS:-"opendde protenix-v2"}

mkdir -p "$OUT/fixtures"
for t in $TARGETS; do
  acc=${t%%:*}; rest=${t#*:}; want=${rest%%:*}; stem=${rest#*:}
  fa="$OUT/fixtures/$stem.fasta"
  if [ ! -s "$fa" ]; then
    curl -fsS "https://rest.uniprot.org/uniprotkb/$acc.fasta" -o "$OUT/fixtures/$acc.raw" || exit 1
    "$PY" - "$acc" "$want" "$fa" <<'EOF' || exit 1
import pathlib, sys
acc, want, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
raw = pathlib.Path(out).parent / (acc + ".raw")
seq = "".join(raw.read_text().splitlines()[1:])
assert len(seq) == want, "%s is %d aa, expected %d" % (acc, len(seq), want)
assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY"), sorted(set(seq) - set("ACDEFGHIKLMNPQRSTVWY"))
pathlib.Path(out).write_text(">A|protein\n%s\n" % seq)
EOF
  fi
  for m in $MODELS; do
    echo "=== $stem $m ==="
    TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_HOLDER=$HOLDER TT_METAL_LOGGER_LEVEL=FATAL \
    PYTHONPATH=$WT "$PY" -u -m tt_bio.main predict "$fa" \
      --model "$m" --out_dir "$OUT/$stem.$m" --single_sequence \
      --recycling_steps 10 --sampling_steps 200 --override || echo "FOLD FAILED: $stem $m"
    cif=$(find "$OUT/$stem.$m" -name "$stem.cif" | head -1)
    res=$(find "$OUT/$stem.$m" -name results.json | head -1)
    [ -z "$cif" ] && continue
    "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));json.dump(d[0],open(sys.argv[2],'w'))" \
      "$res" "$OUT/$stem.$m.conf.json"
    "$PY" "$WT/perf/wh-correctness/check_structure.py" "$cif" --input "$fa" \
      --conf "$OUT/$stem.$m.conf.json" --json "$OUT/$stem.$m.grade.json"
  done
done
