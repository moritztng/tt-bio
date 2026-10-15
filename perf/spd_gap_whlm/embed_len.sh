#!/bin/bash
# One shipped-CLI embed at N residues (ESMC or SaProt) with the size guard off, then check the .npz: [N, d], finite.
# Usage (cwd = tree): embed_len.sh MODEL N OUT [--fast]. Writes OUT/verdict.json; greps the log for the LM FFN refusal.
M=$1 N=$2 O=$3; shift 3
mkdir -p "$O"; rm -rf "$O/npz"
python3 -c "import sys; sys.path.insert(0, 'perf/bhdesign'); from ladder import seq_of; print(f'>L{sys.argv[1]}\n' + seq_of(int(sys.argv[1])))" "$N" > "$O/seq.fasta"
SUB=embed; case $M in saprot*) SUB=saprot;; esac
t0=$(date +%s.%N)
TT_BIO_SIZE_LIMIT=0 python -u -m tt_bio.main $SUB "$O/seq.fasta" --model "$M" --out_dir "$O/npz" --batch_size 1 "$@" > "$O/cli.log" 2>&1; rc=$?
t1=$(date +%s.%N)
python3 - "$O" "$N" "$rc" "$t0" "$t1" <<'PY'
import glob, json, sys
import numpy as np
o, n, rc, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
v = dict(rc=rc, wall_s=round(t1 - t0, 1), refused=open(f"{o}/cli.log").read().count("[lm-ffn] DRAM refused"))
f = sorted(glob.glob(f"{o}/npz/**/*.npz", recursive=True))
if f:
    z = np.load(f[0]); a = max((z[k] for k in z.files), key=lambda a: a.size)
    v.update(npz=f[0], shape=list(a.shape), finite=bool(np.isfinite(a).all()), rows_ok=n in a.shape)
v["pass"] = rc == 0 and v.get("finite", False) and v.get("rows_ok", False)
json.dump(v, open(f"{o}/verdict.json", "w")); print(v)
PY
