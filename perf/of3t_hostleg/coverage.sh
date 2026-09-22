#!/usr/bin/env bash
# of3t-hostleg: the coverage number, re-MEASURED by re-running the census rather than by adding
# 0.75304 to 92.15682 on paper.
#
# `perf/of3t_readable_mass/readable_mass.py` is RE-RUN, never edited: it belongs to a concluded
# row (A33). Its `--compared` input is the union of the published 907-tensor compared set and the
# tensors this row's arm added, built here so the difference between the two runs is exactly the
# set this row changed and nothing else.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
D=/home/ttuser/of3t_hostleg
ARM=${1:-refatom}

"$PY" - "$D/shipped_vs_FLOAT64.json" "$D/device_grads_hl_${ARM}.pt" "$D/compared_${ARM}.json" <<'PYEOF'
import json, sys, torch
pub, dump, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = [e["param"] for e in json.load(open(pub))]
d = torch.load(dump, map_location="meta", weights_only=False, mmap=True)
added = sorted(n for n, v in d.items() if v is not None and n not in set(base))
json.dump([{"param": p} for p in sorted(set(base) | set(added))], open(out, "w"))
print(f"published {len(base)} + added {len(added)} -> {len(set(base) | set(added))}")
print("added:", *added, sep="\n  ")
PYEOF

for tag in BASE "$ARM"; do
  if [ "$tag" = BASE ]; then CMP="$D/shipped_vs_FLOAT64.json"; else CMP="$D/compared_${ARM}.json"; fi
  OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_readable_mass/readable_mass.py \
    --f64 "$D/grads_f64_043.pt" \
    --expect-f64 1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
    --checkpoint "$HOME/of3-weights/of3-p2-155k.pt" \
    --compared "$CMP" \
    --other-boundary "$D/dev_scope_CTRL_c64.pt" \
    --host-applied input_embedder.atom_attn_enc.linear_q.0.weight \
    --host-applied-prefix diffusion_module.atom_attn_enc.ref_atom_feature_embedder. \
    --host-applied-prefix input_embedder.atom_attn_enc.ref_atom_feature_embedder. \
    --out "perf/of3t_hostleg/READABLE_MASS_${tag}.json" || exit 1
  "$PY" -c "
import json,sys
d=json.load(open('perf/of3t_hostleg/READABLE_MASS_${tag}.json'))
h=d['headline']; c=d['classes']
print('${tag}', 'pct_comparable_today', repr(h['pct_comparable_today']), 'n_compared', h['n_compared'])
for k in ('COMPARED','HOST_APPLIED'):
    if k in c: print('   ',k, c[k]['n_tensors'], repr(c[k]['pct_of_model_mass']))
"
done
