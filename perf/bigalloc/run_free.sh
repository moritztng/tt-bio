#!/bin/bash
# run_free.sh TREE OUT ARGS...: run trimul_mask_clash.py from ~/TREE on the first whglx chip whose
# lease is released by a dead holder with no hold note, never a cardblocked one. Exit 3: none free.
tree=$1; shift; out=$1; shift
cd /home/agent/leases || exit 1
c=
for f in *.json; do
  n=$(echo "$f" | sed "s/j10glx02-card//;s/.json//")
  case $n in 1|24|25|26|27) continue;; esac
  python3 - "$f" <<'P' && flock -n "$f" true && { c=$n; break; }
import json, os, sys
d = json.load(open(sys.argv[1])); p = d.get("pid")
sys.exit(0 if d.get("released") and not os.path.exists(f"/proc/{p}") and not d.get("note") else 1)
P
done
[ -n "$c" ] || exit 3
cd ~/"$tree" || exit 1
export PYTHONPATH=$PWD TT_BIO_LEASE_DIR=/home/agent/leases TT_METAL_CACHE=/home/agent/.cache/tt-metal-cache-mgxb
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
echo "card $c $(date -u +%T)"
TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:mgx-bigalloc \
  timeout 1500 ~/env/bin/python perf/bigalloc/trimul_mask_clash.py "$@" > "$out" 2>&1
echo "rc=$?"
grep -E "^S=|l1 unres|Error|\[tt-bio\] trimul" "$out" | cut -c1-400
