#!/bin/bash
# The cross-model spot check for TT_BIO_DEVICE_CONDITIONING, on the tree that ships it on.
#
# The call-site census says no model but Boltz-2 executes a changed branch. Digest reproduction is
# a stronger statement than an RMSD comparison, so this folds the two sibling models on the default
# tree and checks they still write the digests docs/tuning-flags.md quotes, recorded in
# perf/b2z2_msa_census/xmodel/*.json before this flag existed.
#
# One arm, because this is a reproduction and not an A/B. xmodel_pwa_ab.py exits 1 on a one-arm run
# no matter what happens -- its verdict needs both arms -- so its exit code says nothing here and
# the digest is checked directly instead. Re-running is cheap: a model whose JSON is already there
# is not folded again.
set -u
cd /home/ttuser/.coworker/wt/b2z2-cond-ship || exit 1

check() {  # model expected_digest
  python3 - "$1" "$2" <<'PY'
import json, sys
model, want = sys.argv[1], sys.argv[2]
d = json.load(open(f"perf/b2z2_cond/out/xmodel_{model}_512.json"))
got = d["runs"][0]["cif_sha256"]["cdk2x2_512.cif"]
ok = got == want
print(f"{model:12s} {'MATCH' if ok else 'DIFFERS'}  got {got}  want {want}  "
      f"plddt {d['runs'][0]['plddt']}  {d['runs'][0]['fold_s']}s")
sys.exit(0 if ok else 1)
PY
}

rc=0
for spec in protenix-v2:15772214c5b9e990 openfold3:6ee6ac7a3e730688; do
  m=${spec%%:*}; want=${spec##*:}
  out="perf/b2z2_cond/out/xmodel_${m}_512.json"
  if [ ! -f "$out" ]; then
    echo "=== folding $m ==="
    env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:b2z2-cond-ship \
      /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_msa_census/xmodel_pwa_ab.py \
        --model "$m" --size 512 --arms on --out "$out"
  fi
  [ -f "$out" ] || { echo "$m: no output, fold did not run"; rc=1; continue; }
  check "$m" "$want" || rc=1
done
echo "spot check: $([ $rc -eq 0 ] && echo 'both models reproduce their published digest' || echo 'MISMATCH')"
exit $rc
