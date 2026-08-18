#!/usr/bin/env bash
# Publish site/ to both places it is served from:
#   tt-bio.com                     Cloudflare Pages, the canonical site
#   moritztng.github.io/tt-bio/    GitHub Pages, the URL shared before the domain existed
# Run from anywhere. Requires ~/.coworker/cloudflare.env (Pages:Edit token) and gh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -r "$HOME/.coworker/cloudflare.env" ]; then
  echo "missing ~/.coworker/cloudflare.env (CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN)" >&2
  exit 1
fi
set -a; . "$HOME/.coworker/cloudflare.env"; set +a

echo "==> Cloudflare Pages"
npx --yes wrangler@latest pages deploy "$here" \
  --project-name=tt-bio --branch=main --commit-dirty=true

echo "==> GitHub Pages (workflow_dispatch only, so it needs firing)"
gh workflow run pages.yml --repo moritztng/tt-bio --ref main

echo "==> verify"
for u in https://tt-bio.com/ https://tt-bio.com/benchmarks.html; do
  printf '    %-38s %s\n' "$u" "$(curl -sL -o /dev/null -w '%{http_code}' "$u")"
done
