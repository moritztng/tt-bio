#!/usr/bin/env bash
# Fetch RFdiffusion3's 2.51 GB inference checkpoint and prove it is the one every other GPU
# column ran.
#
# gpu_rfd3_setup.sh installs the code and not the weights, and whatever normally prompts for
# them cannot prompt under a detached run: the A100 pass died at its first design with
# "AssertionError: Invalid checkpoint: rfd3" for exactly this reason. So the URL is read out
# of foundry's own registry (never hardcoded here, so a moved artifact fails loudly instead of
# 404-ing into a partial file) and the digest is checked before anything uses it.
set -uo pipefail
DIGEST=9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1
DST=${DST:-/root/.foundry/checkpoints/rfd3_latest.ckpt}
REG=${REG:-/work/fsrc/foundry/inference_engines/checkpoint_registry.py}
mkdir -p "$(dirname "$DST")"

if [ -s "$DST" ] && [ "$(sha256sum "$DST" | cut -d' ' -f1)" = "$DIGEST" ]; then
  echo "checkpoint already present and matches $DIGEST"; exit 0
fi

[ -r "$REG" ] || { echo "no registry at $REG -- did the foundry clone succeed?"; exit 1; }
URL=$(grep -oE 'https?://[^"'"'"' ]*rfd3[^"'"'"' ]*\.ckpt' "$REG" | head -1)
[ -n "$URL" ] || { echo "no rfd3 .ckpt URL in $REG:"; grep -n 'ckpt' "$REG" | head -20; exit 1; }
echo "pulling $URL"
curl -sSL --retry 3 -o "$DST.part" "$URL" || { echo "download failed"; exit 1; }
GOT=$(sha256sum "$DST.part" | cut -d' ' -f1)
if [ "$GOT" != "$DIGEST" ]; then
  echo "DIGEST MISMATCH: got $GOT, want $DIGEST -- refusing to use it"; exit 1
fi
mv "$DST.part" "$DST"
ls -l "$DST"
echo "checkpoint OK $DIGEST"
