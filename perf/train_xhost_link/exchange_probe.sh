#!/bin/sh
# Cross-host gradient-exchange probe. Card-free: it measures the network leg and the host-side
# sum, which is all a data-parallel step exchanges once the gradient is already on the host.
#
# The payload is 28,446,060 B -- ABodyBuilder3's 7,111,515 trainable parameters at float32, the
# figure tt_bio/train/hostreduce.py documents. Random bytes rather than zeros, because a
# compressible payload flatters any link.
#
# Usage: exchange_probe.sh <hostA> <hostB> [reps]
# Run it from a third machine that can ssh to both, or from either host.
set -e
A=${1:?hostA}; B=${2:?hostB}; REPS=${3:-3}
N=28446060
stage() {
  ssh -o BatchMode=yes "ttuser@$1" "f=/tmp/xhost_payload.bin; \
    [ -s \$f ] && [ \"\$(stat -c%s \$f)\" = $N ] || { dd if=/dev/urandom of=\$f bs=1M count=28 status=none; truncate -s $N \$f; }; \
    echo \"\$(hostname) payload \$(stat -c%s \$f) B\""
}
leg() { # $1 from, $2 to, $3 label
  ssh -o BatchMode=yes "ttuser@$1" "/usr/bin/time -f '$3 %e' sh -c \
    'cat /tmp/xhost_payload.bin | ssh -o BatchMode=yes ttuser@$2 \"cat > /dev/null\"' 2>&1 | tail -1"
}
stage "$A"; stage "$B"
echo "--- one way, n=$REPS each ---"
i=0; while [ $i -lt "$REPS" ]; do leg "$A" "$B" "$A->$B"; i=$((i+1)); done
i=0; while [ $i -lt "$REPS" ]; do leg "$B" "$A" "$B->$A"; i=$((i+1)); done
echo "--- both directions at once, n=$REPS: the slower leg is the exchange ---"
i=0; while [ $i -lt "$REPS" ]; do
  { leg "$A" "$B" "$A->$B" & leg "$B" "$A" "$B->$A" & wait; } | sort | tr '\n' ' '; echo
  i=$((i+1))
done
