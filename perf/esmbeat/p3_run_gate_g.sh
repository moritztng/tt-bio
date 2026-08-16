#!/usr/bin/env bash
# Release gate for E + C-in + F + G, on card 0, after the chip-0 reset. A fresh workdir (.gate_g),
# because .gate_cinf holds the first attempt, whose only leg died inside a wedged `ttnn.open_device`
# on a chip left dirty by a SIGKILL.
#
# --workers localhost:0, never qb1:0: parse_workers compares the spec against socket.gethostname()
# and a mismatch ssh-es every device leg to an unresolvable host (21 silent ERROR legs in p2).
#
# pytest runs AFTER the gate, and with test_protenix_largeN::test_fold_512_no_oom deselected. That
# test spun for 2 h 32 min at 92 % CPU on card 0 this pass (py-spy: same Pairformer block index and
# the same frame 60 s apart), ignored SIGINT and SIGTERM, and the SIGKILL that ended it left the
# chip dirty enough to wedge the next device open. Running it unattended would put the whole box
# back in that state; it is recorded as owed against clean origin/main instead. Nothing in this
# branch touches Protenix (SwiGLUFFN is constructed only in ESMC/ESMFold2, §9.8).
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3

echo "=== GATE START $(date -u +%FT%TZ) ==="
$PY -u scripts/full_parity_gate.py --workers localhost:0 --workdir "$WT/.gate_g"
echo "GATE_RC=$? $(date -u +%FT%TZ)"

$PY -m pytest -q --tb=short \
  --deselect tests/test_protenix_largeN.py::test_fold_512_no_oom \
  > "$WT/.gate_g_pytest.log" 2>&1
echo "PYTEST rc=$? $(date -u +%FT%TZ)"
tail -12 "$WT/.gate_g_pytest.log"
echo "CHAIN_G_DONE $(date -u +%FT%TZ)"
