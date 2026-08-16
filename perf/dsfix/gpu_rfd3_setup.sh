#!/bin/bash
# Install both RFD3 arms on a rented GPU box, detached.
#
# Arm A (/work/v_head): upstream foundry `production` @ 4010e3e2e, which is where
#   dense_sdpa_pairbias_attention (#371, default on at inference) lives. Not in any PyPI release.
# Arm B (/work/v_pip):  rc-foundry[rfd3]==0.2.0, what the documented install gives a user today.
#
# rc-foundry needs python 3.12 and the pytorch images ship 3.11, so both arms are 3.12 uv venvs.
# RFD3 also needs a C compiler at RUNTIME (the token initializer shells out to one), which the
# runtime image does not have -- that is what build-essential is for.
#
# Writes /work/SETUP_OK or /work/SETUP_FAIL as the last thing it does. Poll for the marker.
set -u
exec >>/work/setup.log 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="
FAIL() { echo "SETUP FAILED: $*"; echo "$*" > /work/SETUP_FAIL; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential git || FAIL "apt"
echo "--- gcc: $(gcc --version | head -1)"

/opt/conda/bin/pip install -q uv || FAIL "uv"
UV=/opt/conda/bin/uv

# --- arm A: upstream production HEAD -----------------------------------------------------------
cd /work
if [ ! -d fsrc/.git ]; then
  git clone -q https://github.com/RosettaCommons/foundry.git fsrc || FAIL "clone foundry"
fi
cd /work/fsrc
git checkout -q 4010e3e2e || FAIL "checkout 4010e3e2e"
echo "--- foundry HEAD: $(git rev-parse HEAD)"
git show --stat 4010e3e2e | head -20

$UV venv --python 3.12 /work/v_head || FAIL "venv v_head"
if ! $UV pip install -q --python /work/v_head/bin/python -e "/work/fsrc/models/rfd3[rfd3]"; then
  echo "--- editable install of models/rfd3 failed, falling back to 0.2.0 + attention.py overlay"
  $UV pip install -q --python /work/v_head/bin/python "rc-foundry[rfd3]==0.2.0" || FAIL "v_head 0.2.0"
  SP=$(/work/v_head/bin/python -c 'import rfd3,pathlib;print(pathlib.Path(rfd3.__file__).parent)')
  cp /work/fsrc/models/rfd3/src/rfd3/model/layers/attention.py "$SP/model/layers/attention.py" \
     || FAIL "attention overlay"
  echo "OVERLAY" > /work/v_head/INSTALL_ROUTE
else
  echo "EDITABLE" > /work/v_head/INSTALL_ROUTE
fi
echo "--- v_head route: $(cat /work/v_head/INSTALL_ROUTE)"

# --- arm B: the documented pip install ---------------------------------------------------------
$UV venv --python 3.12 /work/v_pip || FAIL "venv v_pip"
$UV pip install -q --python /work/v_pip/bin/python "rc-foundry[rfd3]==0.2.0" || FAIL "v_pip install"

# --- prove the two builds differ in the one function that matters ------------------------------
for V in v_head v_pip; do
  echo "--- $V"
  /work/$V/bin/python - <<'PY'
import inspect, torch, rfd3.model.layers.attention as A
from importlib.metadata import version, PackageNotFoundError
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
print("  rc-foundry", v("rc-foundry"), "torch", torch.__version__,
      "cueq-torch", v("cuequivariance-torch"), "cueq-ops-torch", v("cuequivariance-ops-torch"))
print("  dense_sdpa_pairbias_attention:", hasattr(A, "dense_sdpa_pairbias_attention"))
print("  use_dense_sdpa_pairbias:      ", hasattr(A, "use_dense_sdpa_pairbias"))
print("  sparse_pairbias_attention:    ", hasattr(A, "sparse_pairbias_attention"))
print("  attention.py:", inspect.getsourcefile(A))
PY
done

echo "=== setup ok $(date -u +%FT%TZ) ==="
echo ok > /work/SETUP_OK
