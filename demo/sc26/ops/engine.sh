#!/usr/bin/env bash
# The demo's fold service, as the sc26-engine user unit runs it.
#
# Offline by construction: every model file is already in the local Hugging Face cache, and the
# hub libraries are told never to ask the network. Takes the fleet leases for its chips, then
# execs the server in this same process, so the leases name the service's main pid.
set -euo pipefail
ops=$(cd "$(dirname "$0")" && pwd)
demo=$(dirname "$ops")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:sc26-demo}
py=${SC26_PYTHON:-$HOME/tt-bio-dev/env/bin/python3}
chips=$("$py" "$ops/leases.py" hold "${SC26_CHIPS:-0,1,2,3}" $$)
echo "sc26-engine: chips ${chips:-none} (asked for ${SC26_CHIPS:-0,1,2,3})"
if [ -n "$chips" ]; then
  mode=(--chips "$chips" --reset-cmd "$ops/reset_board.sh")
else
  mode=(--replay-only)
fi
exec "$py" -u "$demo/engine/server.py" "${mode[@]}" --port "${SC26_PORT:-8626}" \
  --logdir "${SC26_LOGDIR:-$HOME/sc26-logs/engine}" --record "${SC26_RECORD:-}" \
  --stall-s 120 --warm-s 300 ${SC26_ENGINE_ARGS:-}
