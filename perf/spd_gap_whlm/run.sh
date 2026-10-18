#!/bin/bash
# One device job on a pool chip: flock the chip, pin it, sample its AICLK (and the host load) once a second for the
# whole run, then run the command. Usage: run.sh CHIP OUT -- cmd args...   (cwd = the tree)
# Under the BH feed the card flock is already held and LOCK names a private one.
# OUT gets aiclk.tsv (epoch, node, MHz, load1), log, rc. The node is the /dev/tenstorrent/N the job itself holds open,
# read from /proc each second: a UMD id is not a node on a Galaxy (.107 maps UMD 0-15 to nodes 16-31), and a map file
# that lacks the chip silently sampled another chip's clock for every run on 10-18 (umd2node.json there is "{}").
CHIP=$1 O=$2; shift 3
[[ $CHIP =~ ^[0-9]+$ ]] || { echo "run.sh: chip '$CHIP' is not a chip number, refusing (an empty pin opens every chip)" >&2; exit 2; }
mkdir -p "$O"
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"; flock -w 600 9 || { echo flockbusy > "$O/rc"; exit 3; }
# Nodes open by $1's process tree (the job); none yet while it loads.
held() {
  local p q kids=$1 all=$1
  while [ -n "$kids" ]; do q=; for p in $kids; do q="$q $(cat /proc/$p/task/*/children 2>/dev/null)"; done; kids=$q; all="$all $q"; done
  for p in $all; do ls -l /proc/$p/fd 2>/dev/null; done | grep -o '/dev/tenstorrent/[0-9]*' | cut -d/ -f4 | sort -un
}
export TT_VISIBLE_DEVICES=$CHIP PYTHONPATH=$PWD
# A tree whose tt_bio is missing (a git archive | tar cut off mid-stream) silently imports the venv's editable
# install instead, i.e. the serving engine's code: ph6/sw6 graded JapanFold's engine as the branch on 10-18.
P=$(python3 -c "import importlib.util as u; print(u.find_spec('tt_bio').origin)" 2>/dev/null)
[[ $P == "$PWD/tt_bio/__init__.py" ]] || { echo "run.sh: tt_bio resolves to '$P', not this tree ($PWD), refusing" | tee "$O/log" >&2; echo "badtree" > "$O/rc"; exit 4; }
"$@" > "$O/log" 2>&1 & J=$!
( while kill -0 $J 2>/dev/null; do for n in $(held $J); do
    echo -e "$(date +%s)\t$n\t$(cat "/sys/class/tenstorrent/tenstorrent!$n/tt_aiclk" 2>/dev/null)\t$(cut -d' ' -f1 /proc/loadavg)"
  done; sleep 1; done ) > "$O/aiclk.tsv" & S=$!
wait $J; rc=$?
kill $S 2>/dev/null; N=$(cut -f2 "$O/aiclk.tsv" | sort -u | tr '\n' ',' | sed 's/,$//')
echo "$rc chip=$CHIP node=${N:-none} $(date -u +%FT%TZ)" > "$O/rc"; exit $rc
