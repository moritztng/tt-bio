#!/usr/bin/env bash
# After make_msa.py: bundle acc.json + the per-complex a3m files for the GPU box and mark ACCURACY-READY in GPU.md.
#   DATA=<data dir> bash prep_box.sh
set -euo pipefail
DATA=${DATA:?}; HERE=$(cd "$(dirname "$0")" && pwd); G=/home/moritz/.coworker/state/pfm/GPU.md
n=$(ls "$DATA"/inputs/*.yaml | wc -l); m=$(cut -f1 "$DATA/msa/depth.tsv" | sort -u | wc -l)
[ "$n" = "$m" ] || { echo "only $m of $n MSAs built"; exit 1; }
rm -rf "$DATA/box"; mkdir -p "$DATA/box/msa"
for d in "$DATA"/msa/*/; do [ "$(basename "$d")" = cache ] || cp -r "$d" "$DATA/box/msa/"; done
~/bcx_hostcut_venv/bin/python "$HERE/make_json.py" "$DATA" /root/pfm/acc/msa > "$DATA/box/acc.json"
cp "$HERE/run_acc.sh" "$DATA/box/"
sz=$(du -sh "$DATA/box" | cut -f1)
grep -q '^ACCURACY-READY:' "$G" || python3 - "$G" "$DATA" "$sz" <<'PY'
import sys; g, data, sz = sys.argv[1:]
s = open(g).read(); anchor = "pfm-gpu copies it to box A"
line = (f"ACCURACY-READY: {__import__('datetime').datetime.utcnow():%Y-%m-%dT%H:%MZ} (pfm-accuracy, auto) {data}/box/ ({sz}: acc.json with 11 items, "
        f"msa/, run_acc.sh) -> box /root/pfm/acc/, then `CFG=a100 bash /root/pfm/acc/run_acc.sh`. Depths: {data}/msa/depth.tsv.\n")
open(g, "w").write(s.replace(anchor, line + anchor, 1))
PY
echo PREP-DONE
