#!/usr/bin/env bash
# Deploy the JapanFold landing page to GitHub Pages (moritztng/japanfold-landing,
# served as https://landing.japanfold.com).
#
# Ships tt_bio/platform/landing/ at the HEAD of this branch — never a dirty
# tree — plus a CNAME file. Idempotent: a no-change run pushes nothing.
#
# One-time setup (already done): create the public repo, then
#   gh api repos/moritztng/japanfold-landing/pages -X POST -f "source[branch]=main" -f "source[path]=/"
# Rollback: point the landing.japanfold.com CNAME back to
#   e3d9384a-ade9-4198-bc17-ebc087bd7168.cfargotunnel.com (proxied); Flask still
#   serves the page unchanged.
set -euo pipefail

BRANCH=aiand-bio-platform
LANDING_DIR=tt_bio/platform/landing
DEST_REPO=moritztng/japanfold-landing

cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain -- "$LANDING_DIR")" ]; then
    echo "refusing: $LANDING_DIR has uncommitted changes" >&2
    exit 1
fi
git fetch --quiet origin "$BRANCH"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
    echo "refusing: HEAD != origin/$BRANCH — pull first, deploys must match the pushed branch" >&2
    exit 1
fi
if [ "$(git config user.email)" != "moritz.thuening@gmail.com" ]; then
    echo "refusing: git identity is not moritztng" >&2
    exit 1
fi
SHA=$(git rev-parse --short HEAD)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Bundle = landing/ at HEAD (git archive: committed bytes only) + CNAME.
git archive HEAD "$LANDING_DIR" | tar -x -C "$TMP"
SITE="$TMP/site"
mv "$TMP/$LANDING_DIR" "$SITE"
echo landing.japanfold.com > "$SITE/CNAME"
touch "$SITE/.nojekyll"

# Gates: no secrets, and every local reference in index.html exists in the bundle.
if find "$SITE" -type f \( -name '*.env' -o -name '*.jsonl' -o -name '*.pem' -o -name '*token*' \) | grep -q .; then
    echo "refusing: secret-shaped file in the bundle" >&2
    exit 1
fi
python3 - "$SITE" <<'PY'
import re, sys, pathlib
site = pathlib.Path(sys.argv[1])
html = (site / "index.html").read_text(encoding="utf-8")
refs = set(re.findall(r'''(?:src|href|poster)=["'](/[^"']*)["']''', html))
refs |= set(re.findall(r'''url\(["']?(/[^"')]+)''', html))
missing = [r for r in sorted(refs) if r != "/" and not (site / r.lstrip("/")).is_file()]
if missing:
    sys.exit("refusing: index.html references missing from bundle: " + ", ".join(missing))
print(f"asset gate: {len(refs)} local references, all present")
PY

git clone --quiet "https://github.com/$DEST_REPO" "$TMP/repo" 2>/dev/null || {
    # First deploy against an empty repo: clone warns, that's fine.
    git clone "https://github.com/$DEST_REPO" "$TMP/repo"
}
rsync -a --delete --exclude=.git --exclude=README.md "$SITE/" "$TMP/repo/"

cd "$TMP/repo"
if [ -z "$(git status --porcelain)" ]; then
    echo "already up to date with $BRANCH@$SHA"
    exit 0
fi
git add -A
git commit --quiet -m "deploy: landing from aiand-bio $BRANCH@$SHA"
git push --quiet origin HEAD
echo "deployed $BRANCH@$SHA -> github.com/$DEST_REPO"
