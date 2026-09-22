# Sourced, not run. The shell half of perf/refpath.py: one place that names the campaign's
# reference trees, and one call that PROVES the tree under test is the tree that resolved.
#
# D153: 43 scripts hard-coded paths under /home/ttuser/of3t_rebase/, which worker.sh removed with
# its row's worktree. A PYTHONPATH entry that does not exist is not an error -- it resolves
# nothing, `import openfold3` falls through to 0.5.0 in pylibs, and the run reports 0.4.3.
# Composing the path correctly is therefore not enough; ref_assert is the part that is evidence.

REF_ROOT=/home/ttuser/of3t-campaign-refs
REF_OF3PKG=$REF_ROOT/of3pkg043
REF_OF3PKG050=$REF_ROOT/of3pkg050
REF_DEPS=/home/ttuser/of3t_gradients/deps
REF_PYLIBS=/home/ttuser/of3t_gradients/pylibs
REF_CODE=/home/ttuser/of3t_gradients/ref
REF_BUNDLE=$REF_ROOT/bundle_min_043
REF_CAP=$REF_ROOT/cap
REF_CAP_LADDER=/home/ttuser/of3t_trunkdepth/cap_ladder
REF_DIFFCAP=/home/ttuser/of3t_softgrad/diffcap043
REF_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ref_pythonpath [extra dirs...] -- the 0.4.3 tree first, then `deps`, then whatever the caller
# names, in the caller's order.
#
# `pylibs` is NOT in the default: on PYTHONPATH it sits ahead of the venv's site-packages, so
# adding it to a script that did not have it could change which torchmetrics or pytorch_lightning
# that script imports. Scripts that had it pass "$REF_PYLIBS" as their first extra, which puts it
# exactly where it was. (`refpath.py` may append both unconditionally -- sys.path.append lands
# behind site-packages and can shadow nothing.)
ref_pythonpath() {
    local pp="$REF_OF3PKG:$REF_DEPS" d
    for d in "$@"; do pp="$pp:$d"; done
    printf '%s' "$pp"
}

# ref_assert <python> [tree] -- import openfold3 under the PYTHONPATH already exported and print
# the tree it came from. Exits the calling script if that is not [tree], default $REF_OF3PKG.
# A deliberate 0.5.0 control passes "$REF_OF3PKG050" and is held to that instead.
ref_assert() {
    "$1" "$REF_SH_DIR/refpath.py" --path-only --pkg "${2:-$REF_OF3PKG}" || exit 1
}

# ref_require <path...> -- refuse up front for any input that is not there.
ref_require() {
    local p miss=""
    for p in "$@"; do [ -e "$p" ] || miss="$miss $p"; done
    if [ -n "$miss" ]; then
        echo "reference inputs missing:$miss -- see perf/refpath.py" >&2
        exit 1
    fi
}
