#!/bin/bash
# Install RF3 on a rented GPU box, detached. Writes /work/SETUP_OK or /work/SETUP_FAIL last;
# poll for the marker.
#
# rc-foundry requires python >= 3.12 and the pytorch images ship 3.11, so this builds a 3.12 uv
# venv. `rc-foundry[rf3]` pulls the cuEquivariance wheels (cu12) that RF3's triangle attention and
# triangle multiplication route to; whether they are actually REACHED is a question for the
# counters in gpu_rf3_run.py, not for this script.
#
# The checkpoint (`ckpt_path: rf3` -> rf3_foundry_01_24_latest_remapped.ckpt) is fetched here
# rather than left to lazy download inside a timed fold.
set -u
exec >>/work/setup.log 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="
FAIL() { echo "SETUP FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential git wget || FAIL "apt"
echo "--- gcc: $(gcc --version | head -1)"

PIP=$(command -v pip3 || command -v pip || echo /opt/conda/bin/pip)
$PIP install -q uv || FAIL "uv"
UV=$(command -v uv || echo /opt/conda/bin/uv)

$UV venv --python 3.12 /work/v_rf3 || FAIL "venv"
PY=/work/v_rf3/bin/python
$UV pip install -q --python $PY "rc-foundry[rf3]" || FAIL "rc-foundry[rf3]"

echo "--- versions"
$PY - <<'PY'
from importlib.metadata import version, PackageNotFoundError
import torch
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
for p in ("rc-foundry","atomworks","cuequivariance-torch","cuequivariance-ops-torch-cu12",
          "cuequivariance-ops-cu12","lightning"):
    print("  %-32s %s" % (p, v(p)))
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import foundry
print("  SHOULD_USE_CUEQUIVARIANCE", foundry.SHOULD_USE_CUEQUIVARIANCE)
import rf3.model.layers.attention as A
print("  attention.py", A.__file__, "cuet", getattr(A, "cuet", None) is not None)
PY
[ $? -eq 0 ] || FAIL "import check"

# --- checkpoint --------------------------------------------------------------------------------
CKPT_DIR=${FOUNDRY_CHECKPOINT_DIRS:-$HOME/.foundry/checkpoints}
mkdir -p "$CKPT_DIR"
CKPT="$CKPT_DIR/rf3_foundry_01_24_latest_remapped.ckpt"
if [ ! -s "$CKPT" ]; then
  wget -q --show-progress -O "$CKPT.part" \
    https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt || FAIL "ckpt wget"
  mv "$CKPT.part" "$CKPT"
fi
echo "--- ckpt $(du -h "$CKPT" | cut -f1)  sha256 $(sha256sum "$CKPT" | cut -c1-16)"

# --- warm the AtomWorks CCD cache OUTSIDE any timed fold ---------------------------------------
# atomworks builds a chemical-component cache on first use; a lazy build inside rep 0 would show
# up as fold time (and `protenix-lazy-ccd-cache-race` measured that build at up to 55 min
# elsewhere in this fleet). Force it now.
$PY - <<'PY' || echo "  (ccd warm best-effort: failed, will be paid in the cold rep)"
import time
t0 = time.time()
from atomworks.io.utils.ccd import get_ccd_info   # noqa: F401
try:
    get_ccd_info("ALA")
except Exception as e:
    print("  ccd probe:", type(e).__name__, e)
print("  ccd warm %.1fs" % (time.time() - t0))
PY

echo "=== setup ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP_OK
