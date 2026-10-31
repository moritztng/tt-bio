#!/bin/bash
# Build the Mol-1 upstream reference env under $ROOT (default ~/mol1-upstream/mol1-ref):
#   $ROOT/Mol-1 (pinned sha, untrusted: only pip reads it), $ROOT/venv (CPU torch), $ROOT/data (MOL_ROOT_DIR).
# Assets come from $DL (the verified D7 download: checkpoint/, common.tar.gz, search_database.tar.gz).
#   bash setup_env.sh [ROOT] [DL]
set -euo pipefail
ROOT=${1:-$HOME/mol1-upstream/mol1-ref}
DL=${2:-$HOME/mol1-upstream/dl}
SHA_CODE=e6e64866ee89640a91cef7ee777a85f1d8be03bb
SHA_CKPT=d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6
mkdir -p "$ROOT/data/checkpoint"; cd "$ROOT"
[ -d Mol-1/.git ] || git clone -q https://github.com/molence/Mol-1 Mol-1
git -C Mol-1 checkout -q $SHA_CODE
# python 3.12 from the system, else uv fetches it (Mol-1 needs >=3.12)
if command -v python3.12 >/dev/null; then
  [ -x venv/bin/python ] || python3.12 -m venv venv
  pip() { venv/bin/pip "$@"; }
else
  UV=$(command -v uv || echo "$HOME/.local/bin/uv")
  [ -x venv/bin/python ] || "$UV" venv -q --python 3.12 venv
  pip() { "$UV" pip "$@" --python venv/bin/python; }
fi
pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
pip install -q -e "$ROOT/Mol-1"
pip freeze > "$ROOT/venv-freeze.txt"
echo "$(sha256sum "$DL/checkpoint/mol-1-base.pt" | cut -d' ' -f1)" | grep -qx $SHA_CKPT
ln -sf "$DL/checkpoint/mol-1-base.pt" data/checkpoint/mol-1-base.pt
[ -e data/.untarred ] || { tar -C data -xzf "$DL/common.tar.gz" && tar -C data -xzf "$DL/search_database.tar.gz" && touch data/.untarred; }
echo ENV_DONE
