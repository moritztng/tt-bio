#!/usr/bin/env bash
# The A/A on forward_device: origin/wk/of3t's file against this branch's, masks unset.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-auxgrad
PY=/home/ttuser/tt-bio-dev/env/bin/python
CAP=/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W/perf/of3t_confidence:$W"
export OMP_NUM_THREADS=8
CARD=${CARD:-1}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=1,$CARD TT_BIO_LEASE_HOLDER=worker:of3t-auxgrad
F=tt_bio/openfold3_confidence.py

cp "$F" /tmp/conf_head.py
git show origin/wk/of3t:$F > /tmp/conf_wkof3t.py
cp /tmp/conf_wkof3t.py "$F"
echo "=== arm A: origin/wk/of3t file $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_auxgrad/aa_device.py /tmp/aa_dev_wkof3t.pt "$CAP"; a=$?
git checkout -- "$F"
cmp -s "$F" /tmp/conf_head.py || { echo "RESTORE FAILED"; exit 9; }
echo "RESTORE OK"
echo "=== arm B: this branch $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_auxgrad/aa_device.py /tmp/aa_dev_head.pt "$CAP"; b=$?
[ $a -eq 0 ] && [ $b -eq 0 ] || { echo "an arm failed: $a $b"; exit 1; }
"$PY" - <<'PY'
import json, torch
from pathlib import Path
A = torch.load("/tmp/aa_dev_wkof3t.pt", map_location="cpu", weights_only=False)
B = torch.load("/tmp/aa_dev_head.pt", map_location="cpu", weights_only=False)
rows, ok = [], True
for k in sorted(A):
    e = bool(torch.equal(A[k], B[k]))
    m = float((A[k].double() - B[k].double()).abs().max())
    ok &= e
    rows.append({"tensor": k, "torch_equal": e, "max_abs_delta": m})
    print(f"  {k:34s} torch.equal {e}  max|d| {m:.3e}")
r = {"entry_point": "OF3ConfidenceHead.forward_device, both masks unset",
     "arm_a": "origin/wk/of3t", "arm_b": "wk/of3t-auxgrad HEAD",
     "n_tensors": len(rows), "all_bit_identical": ok, "rows": rows}
Path("perf/of3t_auxgrad/aa_device.json").write_text(json.dumps(r, indent=1) + "\n")
print("A/A", "PASS" if ok else "FAIL", f"-- {sum(x['torch_equal'] for x in rows)} of {len(rows)} bit-identical")
PY
