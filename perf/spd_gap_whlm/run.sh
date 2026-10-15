#!/bin/bash
# One device job on a pool chip: flock the chip, pin it, sample its AICLK (and the host load) once a second for the
# whole run, then run the command. Usage: run.sh CHIP OUT -- cmd args...   (cwd = the tree)
# Under the BH feed the card flock is already held and LOCK names a private one.
# OUT gets aiclk.tsv (epoch, node, MHz, load1), log, rc. The node is the chip's sysfs node from
# ~/spd/galaxy/umd2node.json on a Galaxy, else the chip number.
CHIP=$1 O=$2; shift 3
mkdir -p "$O"
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"; flock -w 600 9 || { echo flockbusy > "$O/rc"; exit 3; }
N=$(python3 -c "import json,os,sys; print(json.load(open(os.path.expanduser('~/spd/galaxy/umd2node.json'))).get(sys.argv[1], sys.argv[1]))" "$CHIP" 2>/dev/null || echo "$CHIP")
( while :; do echo -e "$(date +%s)\t$N\t$(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk" 2>/dev/null)\t$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) > "$O/aiclk.tsv" & S=$!
export TT_VISIBLE_DEVICES=$CHIP PYTHONPATH=$PWD
"$@" > "$O/log" 2>&1; rc=$?
kill $S; echo "$rc chip=$CHIP node=$N $(date -u +%FT%TZ)" > "$O/rc"; exit $rc
