#!/bin/bash
cd ~/b2pship
export TT_VISIBLE_DEVICES=1 PYTHONPATH=$HOME/b2pship
export INERT_DUMP_DIR=$HOME/b2pship/out/inertdump
mkdir -p "$INERT_DUMP_DIR"
for m in openfold3 boltz2; do
  for n in 544 608; do
    echo "=== INERT model=$m residues=$n $(date -u +%FT%TZ)"
    INERT_TAG=${m}_$n timeout 2700 ~/bcx_e2e_venv/bin/python out/inert_drv.py out/inert_in/p$n.yaml \
      --out_dir out/inert_out_${m}_$n --model $m --override 2>&1 | grep -E "INERT_DUMP|Error|error|FATAL|Traceback|refus" | tail -12
    echo "=== rc=$? $(date -u +%FT%TZ)"
  done
done
echo "=== INERT ALL DONE $(date -u +%FT%TZ)"
