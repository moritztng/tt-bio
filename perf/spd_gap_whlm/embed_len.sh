#!/bin/bash
# One shipped-CLI embed at N residues (ESMC or SaProt) with the size guard off, then check the .npz: [N, d], finite.
# N may be a comma list (512,8192): one record per length in one process, so a short page shape can leave state on the
# device before the long one runs (the CLI embeds short to long).
# Usage (cwd = tree): embed_len.sh MODEL N OUT [--fast]. Writes OUT/verdict.json; greps the log for the LM FFN refusal.
M=$1 N=$2 O=$3; shift 3
mkdir -p "$O"; rm -rf "$O/npz"
python3 -c "import sys; sys.path.insert(0, 'perf/bhdesign'); from ladder import seq_of; print(''.join(f'>L{n}\n{seq_of(int(n))}\n' for n in sys.argv[1].split(',')), end='')" "$N" > "$O/seq.fasta"
SUB=embed; case $M in saprot*) SUB=saprot;; esac
t0=$(date +%s.%N)
TT_BIO_SIZE_LIMIT=0 python -u -m tt_bio.main $SUB "$O/seq.fasta" --model "$M" --out_dir "$O/npz" --batch_size 1 "$@" > "$O/cli.log" 2>&1; rc=$?
t1=$(date +%s.%N)
python3 - "$O" "$N" "$rc" "$t0" "$t1" <<'PY'
import glob, json, sys
import numpy as np
o, ns, rc, t0, t1 = sys.argv[1], [int(n) for n in sys.argv[2].split(',')], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
v = dict(rc=rc, wall_s=round(t1 - t0, 1), refused=open(f"{o}/cli.log").read().count("[lm-ffn] DRAM refused"))
f = sorted(glob.glob(f"{o}/npz/**/*.npz", recursive=True))
shapes = []
for p in f:
    z = np.load(p); a = max((z[k] for k in z.files), key=lambda a: a.size)
    shapes.append((list(a.shape), bool(np.isfinite(a).all())))
v.update(npz=f, shape=[s for s, _ in shapes], finite=bool(shapes) and all(ok for _, ok in shapes),
         rows_ok=all(any({n, n + 2} & set(s) for s, _ in shapes) for n in ns))  # per-residue, with or without BOS/EOS
v["pass"] = rc == 0 and v["finite"] and v["rows_ok"]
json.dump(v, open(f"{o}/verdict.json", "w")); print(v)
PY
