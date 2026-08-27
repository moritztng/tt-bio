# tt-bio.com redirect shim

The Cloudflare Pages project `tt-bio` no longer hosts the site. It hosts this
directory, whose only job is to forward every path to the canonical site at
https://moritztng.github.io/tt-bio/ .

The site itself is published to GitHub Pages by `.github/workflows/pages.yml`
from `site/`. `scripts/deploy_site.sh` does both halves: it publishes the site
and it re-asserts this shim, so a deploy can never quietly turn the redirect
back into a second copy of the site.
