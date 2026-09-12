#!/bin/bash
# Build a shadow TT_METAL_HOME whose only real file is a patched core descriptor.
#
# tt-metal resolves core descriptors as get_root_dir() + "tt_metal/core_descriptors/<file>". The
# stock blackhole_140_arch_eth_dispatch.yaml asks for 14 ethernet dispatch cores; qb2's p300c
# harvests 2 of them (UMD eth mask 0x120), so the open aborts on logical eth core 12. Everything
# else in the tree is symlinked straight back at the installed wheel, so this changes one file for
# one process and touches no shared checkout.
set -euo pipefail
SRC=${1:?src ttnn package root}
DST=${2:?shadow root}
rm -rf "$DST"
mkdir -p "$DST/tt_metal/core_descriptors"
for e in "$SRC"/*; do
  b=$(basename "$e"); [ "$b" = tt_metal ] && continue
  ln -s "$e" "$DST/$b"
done
for e in "$SRC"/tt_metal/*; do
  b=$(basename "$e"); [ "$b" = core_descriptors ] && continue
  ln -s "$e" "$DST/tt_metal/$b"
done
for e in "$SRC"/tt_metal/core_descriptors/*; do
  ln -s "$e" "$DST/tt_metal/core_descriptors/$(basename "$e")"
done
rm -f "$DST/tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml"
python3 - "$SRC/tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml" \
           "$DST/tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
# 14 -> 12 ethernet dispatch cores; logical eth 12 and 13 do not exist on a 2x-eth-harvested p300c.
twelve = "[" + ", ".join(f"[0, {i}]" for i in range(12)) + "]"
text = re.sub(r"\[\[0, 0\](?:, \[0, \d+\])+\]", twelve, text)
open(dst, "w").write(text)
print("patched dispatch_cores to 12 entries:", text.count(twelve), "sites")
PY
echo "shadow root: $DST"
