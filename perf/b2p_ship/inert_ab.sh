#!/bin/bash
cd ~/b2pship
export TT_VISIBLE_DEVICES=0 PYTHONPATH=$HOME/b2pship
export INERT_DUMP_DIR=$HOME/b2pship/out/inertdump
export INERT_TAG=of3_544_padupoff
export TT_BIO_TRIATT_HIFI_PAD_UP=0
mkdir -p "$INERT_DUMP_DIR"
timeout 2400 ~/bcx_e2e_venv/bin/python out/inert_drv.py out/inert_in/p544.yaml --out_dir out/inert_out_openfold3_544_padupoff --model openfold3 --override 2>&1 | grep -E "INERT_DUMP|Error|FATAL|Traceback|refus|Out of Memory" | tail -12
echo "AB_RC=$? $(date -u +%FT%TZ)"
