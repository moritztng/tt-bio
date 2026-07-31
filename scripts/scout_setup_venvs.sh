#!/usr/bin/env bash
# Scout-only: build two fully isolated venvs to A/B ttnn 0.68.0 (the pin) vs 0.75.0.
# NOT part of the product. Never run against the shared checkout.
set -uo pipefail
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-upgrade-scout
BASE=/home/ttuser/.coworker/scout-venvs
export PYTHONNOUSERSITE=1          # ~/.local carries ttnn 0.67.4; it must not leak in
mkdir -p "$BASE"
build () {
  local ver="$1" dir="$BASE/v$2"
  echo "########## building $dir with ttnn==$ver ##########"
  rm -rf "$dir"
  python3 -m venv "$dir" || return 1
  "$dir/bin/pip" install -q --upgrade pip setuptools wheel || return 1
  echo "--- installing tt-bio (-e) ---"
  "$dir/bin/pip" install -e "$WT" 2>&1 | tail -20
  echo "--- forcing ttnn==$ver ---"
  "$dir/bin/pip" install "ttnn==$ver" 2>&1 | tail -10
  echo "--- resolved ---"
  PYTHONNOUSERSITE=1 "$dir/bin/python" -c "import importlib.metadata as m; print(\"ttnn\", m.version(\"ttnn\")); print(\"torch\", m.version(\"torch\"))" 2>&1 | tail -3
}
build 0.68.0 68
build 0.75.0 75
echo "########## SETUP COMPLETE ##########"
