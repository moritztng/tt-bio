#!/usr/bin/env bash
# The lossless render (render.py) to the files people play, encoded on another machine so qb2, which
# runs the booth, does no encoding. Reads the frames over ssh and writes, from one pass:
#   <name>-4k-hevc10.mp4   the master: HEVC Main10, 3840x2160, 60 fps, CRF 14
#   <name>-4k-h264.mp4     the same for any player: H.264 High 8-bit, CRF 16
# Usage: encode.sh <host:frames.mkv> <frames to keep> <out dir> <name>
# The render ends with one frame more than the loop (the cut's check frame); it is dropped here.
set -euo pipefail
src=$1 n=$2 out=$3 name=$4
host=${src%%:*} path=${src#*:}
vf="scale=out_color_matrix=bt709:out_range=tv:flags=spline+accurate_rnd+full_chroma_int"
tag=(-color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv)
mkdir -p "$out"
ssh "$host" "cat $path" | nice ffmpeg -v error -stats -y -i - -frames:v "$n" \
  -map 0:v -vf "$vf,format=yuv420p10le" -c:v libx265 -preset slow -crf 14 \
    -x265-params "log-level=error:keyint=120:min-keyint=120:aq-mode=3" -tag:v hvc1 "${tag[@]}" \
    -movflags +faststart "$out/$name-4k-hevc10.mp4" \
  -map 0:v -vf "$vf,format=yuv420p" -c:v libx264 -preset slow -crf 16 -profile:v high -level 5.2 -g 120 \
    "${tag[@]}" -movflags +faststart "$out/$name-4k-h264.mp4"
