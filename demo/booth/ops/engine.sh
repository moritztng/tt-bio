#!/usr/bin/env bash
# The demo's fold service, as the booth-engine user unit runs it.
#
# Offline by construction: every model file is already in the local Hugging Face cache, and the
# hub libraries are told never to ask the network. Takes the fleet leases for its chips, then
# execs the server in this same process, so the leases name the service's main pid.
set -euo pipefail
ops=$(cd "$(dirname "$0")" && pwd)
demo=$(dirname "$ops")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:booth-demo}
py=${BOOTH_PYTHON:-$HOME/tt-bio-dev/env/bin/python3}
# A power cut or reboot in the middle of a JIT link leaves a 0-byte kernel ELF that tt-metal records
# as built; every later start then TT_THROWs "cannot map elf file" and no chip ever folds (SC26 booth
# box, 11-17). Nothing else compiles on this box, so before the engine starts any empty ELF is dead:
# drop its kernel and let it rebuild.
find "$HOME/.cache/tt-metal-cache-tt-bio" -name '*.elf' -size 0 -printf '%h\n' 2>/dev/null \
  | xargs -r -n1 dirname | sort -u | xargs -r rm -rf || true
chips=$("$py" "$ops/leases.py" hold "${BOOTH_CHIPS-0,1,2,3}" $$)
echo "booth-engine: chips ${chips:-none} (asked for ${BOOTH_CHIPS-0,1,2,3})"
if [ -n "$chips" ]; then
  mode=(--chips "$chips" --reset-cmd "$ops/reset_board.sh")
else
  mode=(--replay-only)
fi
exec "$py" -u "$demo/engine/server.py" "${mode[@]}" --port "${BOOTH_PORT:-8626}" \
  --logdir "${BOOTH_LOGDIR:-$HOME/booth-logs/engine}" --record "${BOOTH_RECORD:-}" \
  --stall-s 120 --warm-s 300 ${BOOTH_ENGINE_ARGS:-}
