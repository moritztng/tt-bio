#!/usr/bin/env bash
# Zero-notice acceptance probe for the JapanFold API.
#
# The competition publishes each target only when its window opens, so the service has to accept
# an arbitrary target with no registration, no per-target preparation and no warm-up. That is a
# platform property a deploy can break silently, so this probe belongs in the deploy gate.
#
# Three checks, all against a target generated from a fixed seed and submitted nowhere before:
#   1. fold an unseen two-chain complex                     (MSA off, so it is a pure device path)
#   2. fold an unseen single chain on default params        (exercises the default MSA route)
#   3. design against the structure check 1 just produced   (a target that is seconds old)
#
# One caveat on re-runs: the seed is fixed, so after the first run the MSA for check 2's sequence
# is cached on the host and check 2 stops measuring the cold MSA path (measured: 56 s cold, 24 s
# warm). Availability is still what this gate asserts, and that is unaffected. Pass a different
# seed if you want the cold number back.
#
# Usage: JAPANFOLD_API_KEY=... ./probe.sh [api-base]
set -euo pipefail

API="${1:-https://dev-api-japanfold.aiand.com}"
: "${JAPANFOLD_API_KEY:?set JAPANFOLD_API_KEY}"
AUTH=(-H "Authorization: Bearer ${JAPANFOLD_API_KEY}" -H 'Content-Type: application/json')
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# A sequence nobody has seen. Seeded, so the probe is reproducible and the target is still novel
# to the service: the point is that no allowlist and no cache can have been primed for it.
python3 - "$TMP" <<'PY'
import random, sys, pathlib
random.seed(20260928)
AA = "ACDEFGHIKLMNPQRSTVWY"
d = pathlib.Path(sys.argv[1])
(d / "target.txt").write_text("".join(random.choice(AA) for _ in range(96)))
(d / "binder.txt").write_text("".join(random.choice(AA) for _ in range(56)))
PY
TGT="$(cat "$TMP/target.txt")"; BND="$(cat "$TMP/binder.txt")"

submit() {  # submit <path> <json-file> -> job id
  curl -sf -m 60 -X POST "$API$1" "${AUTH[@]}" --data @"$2" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'
}

await() {  # await <job id> -> final status, prints elapsed seconds
  local id="$1" t0 st
  t0=$(date +%s)
  for _ in $(seq 1 90); do
    st=$(curl -sf -m 30 "${AUTH[@]}" "$API/v1/jobs/$id" \
         | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    case "$st" in succeeded|failed|cancelled) break ;; esac
    sleep 5
  done
  echo "  job $id -> $st in $(( $(date +%s) - t0 ))s"
  [ "$st" = succeeded ]
}

echo "== 1. unseen two-chain complex"
python3 -c 'import json,sys; print(json.dumps({"model":"boltz2","name":"zeronotice-complex",
  "input":">A|protein|empty\n%s\n>B|protein|empty\n%s\n" % (sys.argv[1], sys.argv[2])}))' \
  "$TGT" "$BND" > "$TMP/req1.json"
J1=$(submit /v1/predictions "$TMP/req1.json"); await "$J1"
curl -sf -m 30 "${AUTH[@]}" "$API/v1/jobs/$J1/artifacts/target_1.cif" -o "$TMP/target.cif"
test -s "$TMP/target.cif"

echo "== 2. unseen single chain, default params"
python3 -c 'import json,sys; print(json.dumps({"model":"boltz2","name":"zeronotice-msa","sequence":sys.argv[1]}))' \
  "$TGT" > "$TMP/req2.json"
J2=$(submit /v1/predictions "$TMP/req2.json"); await "$J2"

echo "== 3. design against a structure that is seconds old"
python3 -c 'import json,sys; print(json.dumps({"protocol":"pxdesign-binder","name":"zeronotice-design",
  "structure":open(sys.argv[1]).read(),"chains":"A","binder_length":60,"params":{"num_designs":1}}))' \
  "$TMP/target.cif" > "$TMP/req3.json"
J3=$(submit /v1/designs "$TMP/req3.json"); await "$J3"

echo "zero-notice acceptance: OK"
