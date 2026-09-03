#!/usr/bin/env bash
# Publish the site.
#
# The canonical site is GitHub Pages at https://moritztng.github.io/tt-bio/ ,
# built from site/ by .github/workflows/pages.yml. tt-bio.com is a Cloudflare
# Pages project that now serves only site-redirect/, which forwards every path
# there. Both halves run here, because a deploy that published the site to
# Cloudflare would silently turn the redirect back into a second public copy.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A non-login shell (cron, a fleet worker, an agent turn) does not source the profile that
# puts gh and npx on PATH, and this script died twice mid-deploy that way: Pages triggered,
# the Cloudflare redirect not re-asserted. Resolve both before the first step runs.
command -v gh >/dev/null 2>&1 || PATH="$HOME/.local/bin:$PATH"
if ! command -v npx >/dev/null 2>&1; then
  nodebin="$(ls -d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | sort -V | tail -1)"
  if [ -n "$nodebin" ]; then PATH="$nodebin:$PATH"; fi
fi
export PATH
for t in gh npx; do
  command -v "$t" >/dev/null 2>&1 || { echo "$t not found on PATH; deploy would half-run" >&2; exit 1; }
done

echo "==> asset stamps"
python3 "$root/scripts/stamp_assets.py" --check

echo "==> publish guard"
python3 "$root/scripts/site_publish_guard.py"

echo "==> GitHub Pages (canonical)"
gh workflow run pages.yml --repo moritztng/tt-bio --ref main

echo "==> Cloudflare: re-assert the tt-bio.com redirect"
if [ ! -r "$HOME/.coworker/cloudflare.env" ]; then
  echo "missing ~/.coworker/cloudflare.env (CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN)" >&2
  exit 1
fi
set -a; . "$HOME/.coworker/cloudflare.env"; set +a
npx --yes wrangler@latest pages deploy "$root/site-redirect" \
  --project-name=tt-bio --branch=main --commit-dirty=true

echo "==> verify"
printf '    %-46s %s\n' "moritztng.github.io/tt-bio/" \
  "$(curl -s -o /dev/null -w 'http=%{http_code}' https://moritztng.github.io/tt-bio/)"
printf '    %-46s %s\n' "tt-bio.com (expect 302 -> Pages)" \
  "$(curl -s -o /dev/null -w 'http=%{http_code} -> %{redirect_url}' https://tt-bio.com/)"
