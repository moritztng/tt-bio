#!/bin/bash
# of3t-widthattr deliverable 3: the FLOOR's width curve at 128 and 256, which are reachable
# CPU-side. There is no device arm at those widths -- this row holds no card and of3t-padshape's
# banked intermediate arms (/tmp/of3t/of3t-padshape) no longer exist on either box. So this is a
# floor curve with no arm beside it and it is labelled one.
#
# The boundaries are CROPS of boundary_n384.pt, built the same way of3t-padshape built its
# sweep, and the crop identity of boundary_c64.pt against boundary_n384.pt is measured in
# GROWTH.json rather than assumed.
set -uo pipefail
D=/home/ttuser/of3t_frame384
W=/home/ttuser/of3t_widthattr
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd $W
$PY - <<'PY'
import torch, hashlib, json
b = torch.load("/home/ttuser/of3t_frame384/boundary_n384.pt", map_location="cpu", weights_only=False)
rep = {}
for w in (128, 256):
    o = dict(b)
    o["s_in"] = b["s_in"][:, :w].contiguous()
    o["z_in"] = b["z_in"][:, :w, :w].contiguous()
    o["single_mask"] = b["single_mask"][..., :w].contiguous()
    o["pair_mask"] = b["pair_mask"][..., :w, :w].contiguous()
    p = f"/home/ttuser/of3t_widthattr/boundary_w{w}.pt"
    torch.save(o, p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    m = o["single_mask"].reshape(-1) > 0
    rep[w] = {"path": p, "sha256": h, "real": int(m.sum()),
              "s_in_norm": float(o["s_in"].double().norm()),
              "z_in_norm": float(o["z_in"].double().norm()),
              "s_real_norm": float(o["s_in"].double()[:, m].norm()),
              "z_real_norm": float(o["z_in"].double()[:, m][:, :, m].norm())}
json.dump(rep, open("/home/ttuser/of3t_widthattr/BOUNDARIES.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
PY
export PYTHONPATH=$D/ref:$D/deps
run() {
  local nm=$1 pol=$2 w=$3
  echo "=== $nm $(date -u +%FT%TZ) ==="
  OMP_NUM_THREADS=16 $PY $D/ref_grad.py --tree $D/of3pkg043 \
    --boundary $W/boundary_w${w}.pt --cap-last $D/block47_boundary.pt \
    --policy $pol --crop $w --threads 16 --checkpoint \
    --out $W/${nm}.pt --report $W/${nm^^}.json 2>&1 \
    | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)"
  echo "=== $nm exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
}
for w in 128 256; do
  run ref_f64_w$w      f64      $w
  run ref_bf16auto_w$w bf16auto $w
done
echo "LADDER DONE $(date -u +%FT%TZ)"
