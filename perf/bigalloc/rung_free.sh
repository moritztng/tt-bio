#!/bin/bash
# rung_free.sh MODEL RUNG|FIXTURE.yaml [OUT.jsonl]: wait for a whglx chip whose lease a dead holder
# released with no hold note (never a cardblocked one), then fold MODEL once on it at cap 2.
# A number is a size-ladder rung, folded through release_gate's census fold (the fold mgx-speed's
# time_rungs.py times), JSON on stdout. A yaml goes through perf/whceil/ladder.py into OUT.jsonl
# (default perf/bigalloc/ref.jsonl). Coverage and accuracy folds, not timings. FOLD_TIMEOUT_S
# overrides the census fold's 1800 s, which a loaded box can outrun.
model=$1; rung=$2; out=${3:-perf/bigalloc/ref.jsonl}; tree=$(cd "$(dirname "$0")/../.." && pwd)
pick() {
  cd /home/agent/leases || return 1
  for f in *.json; do
    n=$(echo "$f" | sed "s/j10glx02-card//;s/.json//")
    case $n in 1|24|25|26|27) continue;; esac
    python3 - "$f" <<'P' && flock -n "$f" true && { echo "$n"; return 0; }
import json, os, sys
d = json.load(open(sys.argv[1])); p = d.get("pid")
sys.exit(0 if d.get("released") and not os.path.exists(f"/proc/{p}") and not d.get("note") else 1)
P
  done
  return 1
}
for _ in $(seq 1 240); do
  c=$(pick) && break
  sleep 15
done
[ -n "$c" ] || { echo "no chip"; exit 3; }
cd "$tree" || exit 1
echo "$model $rung card $c $(date -u +%T)"
export PYTHONPATH=$tree TT_BIO_LEASE_DIR=/home/agent/leases TT_METAL_CACHE=/home/agent/.cache/tt-metal-cache-mgxb
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:mgx-bigalloc
case $rung in
  *.yaml) exec ~/env/bin/python perf/whceil/ladder.py --model "$model" --device "$c" --rungs "$rung" \
            --out "$out" --out-root perf/bigalloc/runs --timeout 9000 --env TT_BIO_SIZE_LIMIT=0 \
            --env TT_BIO_LEASE_HOLDER=worker:mgx-bigalloc -- --host_threads 2 ;;
esac
exec ~/env/bin/python - "$model" "$rung" <<'P'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
os.environ["TT_BIO_SIZE_LIMIT"] = "0"
import release_gate as rg
run_fold = rg._run_fold  # this tree's census fold passes no cap: add it to predict
rg._run_fold = lambda cmd, *a, **k: run_fold(cmd + ["--host_threads", "2"] * ("predict" in cmd), *a, **k)
rg.FOLD_TIMEOUT_S = int(os.environ.get("FOLD_TIMEOUT_S", rg.FOLD_TIMEOUT_S))
model, rung = sys.argv[1], int(sys.argv[2])
r = rg._run_census_fold(model, rung, Path("perf/bigalloc/rung") / model, "cover")
print(json.dumps({"model": model, "rung": rung, "card": os.environ["TT_VISIBLE_DEVICES"],
                  **{k: r.get(k) for k in ("error", "refused", "runtime_s", "aiclk", "load",
                                           "structure")}}, default=str), flush=True)
P
