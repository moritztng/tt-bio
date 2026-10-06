#!/usr/bin/env bash
# One loop, start to finish on qb2: build the session for a period of N frames, confirm the cut with a timing
# run, then render the loop at 4K into a lossless file. Usage: pipeline.sh <session dir> <N frames>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); ses=$1; N=$2
P=$(python3 -c "print($N/60)")
python3 "$here/compose.py" build --out "$ses" --period "$P"
python3 "$here/render.py" --session "$ses" --size 1280x720 --end $((N * 4)) > "$ses/timing.log" 2>&1
cp "$ses/slots.jsonl" "$ses/slots-timing.jsonl"
read -r _ off ka kb < <(python3 "$here/compose.py" seam --out "$ses" --warm 2.2 | tee "$ses/seam.txt" | grep ^BEST)
echo "seam: off $off, $ka -> $kb"
[ "$off" = 0 ] || { echo "the cut is not exactly one period; stop"; exit 1; }
nice -n 5 python3 "$here/render.py" --session "$ses" --size 3840x2160 --from "$ka" --to "$kb" --end "$kb" \
  --out "$ses/frames.mkv" > "$ses/render4k.log" 2>&1
cp "$ses/slots.jsonl" "$ses/slots-4k.jsonl"
echo "rendered $ka..$kb"
