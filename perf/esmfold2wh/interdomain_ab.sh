#!/bin/bash
# Does bounding the MSA to 5120 rows move DOMAINS relative to each other?
#
# Fixture: E. coli lysine/arginine/ornithine-binding protein, PDB 2LAO chain A, 238 aa,
# 1.8 A. A natural single chain whose two lobes are sequence-discontinuous (1-89 + 192-238
# and 90-191) and joined by a hinge -- the most sensitive instrument available for
# inter-domain placement, and the opposite of cdk2_640, whose chimeric linker saturated a
# global superposition (state/japanfold-esmfold2-wh-msa-cap-p2.md S21).
#
# 238 aa, not the 600-700 aa the brief asked for, because the 8192-deep arm has to run:
# on qb1's Blackhole the MSA transition asks L*8192*512*2 B in one block and 625 aa OOMs
# on a 5,242,880,000 B request (run_full.log). The row blocking that makes a deep MSA fit
# is small-grid gated, so it is off here, and the fleet's only Wormhole part is the live
# JapanFold Galaxy. L <= ~310 aa is what a full-depth arm fits in on this card.
# The perturbation under test -- 8192 rows vs 5120, the deepest ratio the cap ever
# imposes -- is identical at any length.
set -u
WT=/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p3-prove
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=/home/ttuser/msacap_p3
cd "$WT" || exit 1

PYTHONPATH=$WT $PY - <<'PY'
import hashlib, pathlib, time
from tt_bio.main import _generate_esmfold2_a3m
seq = "".join(l.strip() for l in open("/home/ttuser/msacap_p3/lao_2lao.fasta") if not l.startswith(">"))
h = hashlib.sha256(seq.encode()).hexdigest()[:16]
d = pathlib.Path("/home/ttuser/msacap_p3/msa")
for i in range(12):
    if (d / f"{h}.a3m").exists():
        break
    try:
        _generate_esmfold2_a3m({h: seq}, "lao", d, None, True,
                               "https://api.colabfold.com", None, None, None, None)
    except Exception as e:
        print("msa attempt", i, type(e).__name__, str(e)[:100], flush=True)
        time.sleep(45)
p = d / f"{h}.a3m"
print("a3m rows:", sum(1 for l in p.read_text().splitlines() if l.startswith(">")) if p.exists() else "MISSING")
PY

arm() {  # $1 = tag, $2 = --max_msa_seqs
  TT_VISIBLE_DEVICES=2 TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p3-prove \
  PYTHONPATH=$WT timeout 4000 $PY -m tt_bio.main predict $D/lao_2lao.fasta \
    --model esmfold2 --fast --msa_dir $D/msa --seed 0 --max_msa_seqs "$2" \
    --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
    --out_dir "$D/out/$1" --override > "$D/run_$1.log" 2>&1
  echo "$1 exit=$? cif=$(ls $D/out/$1/*/structures/*.cif 2>/dev/null | wc -l)"
}
arm lao_full 8192
arm lao_cap5120 5120
sha256sum $D/out/lao_full/*/structures/*.cif $D/out/lao_cap5120/*/structures/*.cif

$PY $WT/perf/esmfold2wh/domain_placement_ab.py $D/pdb/2LAO.cif \
  full=$D/out/lao_full/*/structures/*.cif cap5120=$D/out/lao_cap5120/*/structures/*.cif \
  $WT/perf/esmfold2wh/lao_interdomain_ab.json
