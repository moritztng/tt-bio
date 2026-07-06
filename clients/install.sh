#!/bin/sh
# JapanFold CLI installer.
#
#   curl -fsSL https://install.japanfold.com/install.sh | sh
#
# Installs the dependency-free `japanfold` CLI. Prefers pipx (isolated), falls
# back to `pip install --user`. Set JAPANFOLD_CLI_SOURCE to install from a local
# checkout or a git URL instead of PyPI (used in development).
set -eu

SOURCE="${JAPANFOLD_CLI_SOURCE:-japanfold}"

say() { printf '%s\n' "$*" >&2; }

if command -v pipx >/dev/null 2>&1; then
    say "Installing japanfold with pipx…"
    pipx install --force "$SOURCE"
elif command -v python3 >/dev/null 2>&1; then
    say "pipx not found; installing with pip --user…"
    python3 -m pip install --user --upgrade "$SOURCE"
else
    say "error: need python3 (and ideally pipx) on PATH."
    exit 1
fi

say ""
say "Installed. Next:"
say "  export JAPANFOLD_API_KEY=jf_live_...   # from https://japanfold.com/account"
say "  japanfold models"
say "  japanfold predict --sequence MKTAYIAK... --wait --out ./out"
