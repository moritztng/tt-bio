#!/usr/bin/env bash
# Swap the cuEquivariance ops wheels from the cu12 build to the cu13 build, in the two venvs
# whose models are eligible for the Blackwell triangle-attention kernel.
#
# Why this is a real question and not a version fetish: cuEquivariance ships
# "Blackwell-optimized BF16/FP16 forward and backward kernels for cuet.triangle_attention"
# (0.8.0) and a faster sm100f CC 10.0/10.3 forward kernel for hidden_dim <= 256 (0.10.0), and
# the changelog says both are "only available on cu13 builds". boltz-2 and boltzgen run bf16
# and were both published on the cu12 wheel, so on a B200 they may have been measured with a
# shipped Blackwell path installed-out. torch in these venvs is +cu130, so the cu13 wheel is
# also the build that MATCHES their torch; the cu12 wheel is the mismatch.
#
# This produces an ALTERNATE arm. It never replaces a published cell: both numbers get
# reported and the decision is handed up.
set -uo pipefail
V=${1:-}
VENVS=${V:-"/root/venv-boltz /work/venv-bgg"}
OUT=${OUT:-/root/results}
mkdir -p "$OUT"

for venv in $VENVS; do
  PIP="$venv/bin/pip"; PY="$venv/bin/python"
  [ -x "$PIP" ] || { echo "skip $venv (no pip)"; continue; }
  echo "=== $venv"
  "$PIP" freeze 2>/dev/null | grep -i cuequivariance > "$OUT/cueq_before_$(basename "$venv").txt"
  cat "$OUT/cueq_before_$(basename "$venv").txt"
  CU=$("$PY" -c 'import torch;print(torch.version.cuda or "")' 2>/dev/null)
  case "$CU" in
    13*) ;;
    *) echo "REFUSING: torch cuda is '$CU', not 13.x -- a cu13 ops wheel against a cu12 torch \
is a different question than the one being asked"; continue ;;
  esac
  VER=$("$PY" -c 'from importlib.metadata import version;print(version("cuequivariance-torch"))' \
        2>/dev/null || echo 0.11.1)
  "$PIP" uninstall -y -q cuequivariance-ops-torch-cu12 cuequivariance-ops-cu12 2>&1 | tail -2
  "$PIP" install --no-cache-dir -q "cuequivariance-ops-torch-cu13==$VER" 2>&1 | tail -3
  "$PIP" freeze 2>/dev/null | grep -i cuequivariance > "$OUT/cueq_after_$(basename "$venv").txt"
  cat "$OUT/cueq_after_$(basename "$venv").txt"
  # A wheel that installs but cannot import would silently send the model down the torch
  # fallback and the arm would read as "Blackwell kernels are no faster".
  "$PY" - <<'PY'
import torch
import cuequivariance_torch as cuet
print("cueq import OK; torch", torch.__version__, "cap", torch.cuda.get_device_capability())
from importlib.metadata import version
for p in ("cuequivariance-torch", "cuequivariance-ops-torch-cu12", "cuequivariance-ops-torch-cu13"):
    try:
        print("  ", p, version(p))
    except Exception:
        print("  ", p, "absent")
PY
done
