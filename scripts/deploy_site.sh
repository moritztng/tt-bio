#!/usr/bin/env bash
# Publish site/ to both places it is served from:
#   tt-bio.com                     Cloudflare Pages, the canonical site
#   moritztng.github.io/tt-bio/    GitHub Pages, the URL shared before the domain existed
# Run from anywhere. Requires ~/.coworker/cloudflare.env (Pages:Edit token) and gh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../site" && pwd)"

if [ ! -r "$HOME/.coworker/cloudflare.env" ]; then
  echo "missing ~/.coworker/cloudflare.env (CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN)" >&2
  exit 1
fi
set -a; . "$HOME/.coworker/cloudflare.env"; set +a

# A model reaches tt-bio.com only when every processor column is measured. A half-measured row
# still draws its bars, so this refuses the deploy rather than publishing a partial claim.
echo "==> asset stamps"
python3 "$(dirname "${BASH_SOURCE[0]}")/stamp_assets.py" --check
echo "==> publish guard"
python3 "$(dirname "${BASH_SOURCE[0]}")/site_publish_guard.py"

echo "==> Cloudflare Pages"
npx --yes wrangler@latest pages deploy "$here" \
  --project-name=tt-bio --branch=main --commit-dirty=true

echo "==> GitHub Pages (workflow_dispatch only, so it needs firing)"
gh workflow run pages.yml --repo moritztng/tt-bio --ref main

echo "==> verify"
# resolve against a public resolver: a machine that saw NXDOMAIN before the
# record existed can hold a negative cache entry for hours and report a false 000
ip="$(dig +short tt-bio.com @1.1.1.1 | grep -m1 -E '^[0-9.]+$')"
for u in / /benchmarks/ /assets/site.css /data/perf-512aa.json; do
  printf '    %-24s %s\n' "$u" \
    "$(curl -sL --resolve "tt-bio.com:443:$ip" -o /dev/null -w 'http=%{http_code} ttfb=%{time_starttransfer}s' "https://tt-bio.com$u")"
done
