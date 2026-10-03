#!/usr/bin/env bash
# Frame-rate matrix on this box's own GPU. Each line: size, query, then the page's measurement.
#   bench.sh <traj> <seconds> <WxH> "<extra query>" [<WxH> "<extra query>" ...]
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
traj=$1 secs=$2; shift 2
while [ $# -ge 2 ]; do
  sz=$1 extra=$2; shift 2
  out=$(timeout $((secs + 200)) "$here/look.sh" bench "$sz" "traj=$traj&speed=0.5&bench=$secs&$extra" "$secs" 2>&1 | tail -1)
  python3 - "$sz" "$extra" "$out" <<'PY'
import json, sys
sz, extra, line = sys.argv[1:4]
try:
    d = json.loads(line)
except Exception:
    print(sz, extra, "NO RESULT:", line[:200]); sys.exit()
ph = " ".join(f"{k}:p50={v['p50']:.1f}/p95={v['p95']:.1f}ms" for k, v in d.get("phases", {}).items())
print(f"{sz} scale={d['scale']} msaa={d['msaa']} {extra or '-'}: {d['fps']:.1f} fps p50={d['p50']:.1f} p95={d['p95']:.1f} "
      f"p99={d['p99']:.1f} worst={d['worst']:.1f} ms >20ms={d['over20ms']}/{d['frames']} | {ph} | mesh p50={d.get('meshP50')} max={d.get('meshMax')} ms {d.get('ntri')} tris | gpu p50={d.get('gpu50')} max={d.get('gpuMax')} ms n={d.get('gpuN')}")
PY
done
