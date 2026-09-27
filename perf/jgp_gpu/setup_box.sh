#!/usr/bin/env bash
# Route C of the kit's protenix_v2/STOCK.md 'Stack', verbatim steps, on the rented box.
# Stock protenix 2.0.0 (the pinned wheel) on the kit's pinned stack; the same env runs --mode off/exact/fast.
set -euo pipefail
cd /root/kit/protenix_v2
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq ca-certificates wget git build-essential ninja-build kalign rsync >/dev/null
if ! command -v uv >/dev/null; then
  t=$(mktemp) && wget -qO "$t" https://astral.sh/uv/0.12.15/install.sh \
    && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t"
fi
. "$HOME/.local/bin/env"
[ -x venv/bin/python ] || uv venv --seed --managed-python --python 3.11.5 venv
. venv/bin/activate
pip install -q $(grep -E '^pip==' environment/requirements.lock)
grep -v -E '^(#|protenix==)' environment/requirements.lock > /tmp/stack.txt
pip install -q --no-deps -r /tmp/stack.txt
pip install -q --no-deps stock/protenix-2.0.0-py3-none-any.whl
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0
python -c "import protenix.model.layer_norm.layer_norm; print('fastln built')"
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
echo SETUP-DONE
