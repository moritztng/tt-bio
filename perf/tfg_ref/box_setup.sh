#!/usr/bin/env bash
# One-time setup of a rented CUDA box for the TFG reference: upstream OpenDDE v1.2.0, its runtime data, the ABAG
# checkpoint (sha256-verified by upstream's manifest), and a separate DockQ + gemmi venv for scoring.
set -euo pipefail
export OPENDDE_ROOT_DIR=/root/opendde_data
cd /root
[ -d OpenDDE ] || git clone -q --branch v1.2.0 --depth 1 https://github.com/aurekaresearch/OpenDDE.git
cd OpenDDE
# The [gpu] extra is upstream's documented GPU install: cuEquivariance triangle kernels (auto picks them) and Triton.
pip install -q -e ".[gpu]"
# DockQ's pins clash with upstream's, so scoring gets its own venv.
[ -x /root/dq/bin/python ] || { python3 -m venv /root/dq && /root/dq/bin/pip install -q 'DockQ==2.1.3' gemmi numpy; }
bash scripts/download_opendde_data.sh --root "$OPENDDE_ROOT_DIR" --skip-search-database --model-name opendde_v1 --checkpoint opendde_abag.pt
python3 -c "import torch, triton, opendde; print('torch', torch.__version__, 'triton', triton.__version__, 'opendde', opendde.__version__, torch.cuda.get_device_name(0))"
sha256sum "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
