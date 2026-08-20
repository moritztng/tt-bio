#!/bin/bash
# Parallel-range fetch for PXDesign release data.
#
# The upstream origin (pxdesign.tos-cn-beijing.volces.com) serves a single stream at
# ~20 KB/s from outside CN, which is 8 hours for the 556 MB pxdesign checkpoint. It does
# honour Range requests, so N concurrent ranges get the full pipe. aria2c is not installed
# on the QuietBoxes, hence curl. Same problem the GPU-reference task hit and solved with
# aria2c -x16.
#
# Usage: fetch_release_data.sh <dest_dir> [name ...]     (default: all checkpoints)
set -u
DEST="${1:?usage: fetch_release_data.sh <dest_dir> [name ...]}"; shift || true
BASE=https://pxdesign.tos-cn-beijing.volces.com
N=16

declare -A URLS=(
  [pxdesign_v0.1.0]="$BASE/release_model/pxdesign_v0.1.0.pt"
  [protenix_base_default_v0.5.0]="$BASE/release_model/protenix_base_default_v0.5.0.pt"
  [protenix_mini_default_v0.5.0]="$BASE/release_model/protenix_mini_default_v0.5.0.pt"
  [protenix_mini_tmpl_v0.5.0]="$BASE/release_model/protenix_mini_tmpl_v0.5.0.pt"
)
NAMES=("$@"); [ ${#NAMES[@]} -eq 0 ] && NAMES=("${!URLS[@]}")

mkdir -p "$DEST"
for name in "${NAMES[@]}"; do
  url="${URLS[$name]:?unknown name $name}"
  out="$DEST/$name.pt"
  total=$(curl -sI "$url" | tr -d "\r" | awk "/^[Cc]ontent-[Ll]ength:/{print \$2}")
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$total" ]; then
    echo "[skip] $name already complete ($total bytes)"; continue
  fi
  echo "[fetch] $name  $total bytes  in $N ranges"
  chunk=$(( (total + N - 1) / N ))
  pids=()
  for ((i=0;i<N;i++)); do
    s=$(( i * chunk )); e=$(( s + chunk - 1 )); [ $e -ge $total ] && e=$(( total - 1 ))
    [ $s -gt $e ] && continue
    curl -s --retry 20 --retry-delay 5 --retry-all-errors -C - -r "$s-$e" "$url" -o "$(printf "%s.part%03d" "$out" "$i")" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  # Reassemble in NUMERIC order. Zero-padded part names are load-bearing: a bare
  # "part$i" glob sorts lexicographically, so part10..part15 concatenate before part2 and
  # the result is a byte-complete but scrambled file -- the size check below still passes.
  cat $(ls "$out".part* | sort) > "$out" && rm -f "$out".part*
  got=$(stat -c%s "$out")
  if [ "$got" != "$total" ]; then echo "[FAIL] $name got $got want $total" >&2; exit 1; fi
  # Size alone does not prove a correct reassembly. Assert the file actually opens.
  if ! python3 -c "import sys,torch; torch.load(sys.argv[1], map_location=\"cpu\", weights_only=False)" "$out" 2>/dev/null; then
    echo "[FAIL] $name is $got bytes but torch.load failed -- corrupt reassembly" >&2; exit 1
  fi
  echo "[done] $name $got bytes, torch.load ok"
done
echo "[all done]"
