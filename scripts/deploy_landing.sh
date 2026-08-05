#!/usr/bin/env bash
# Deploy the JapanFold landing page to Cloudflare Pages (project japanfold-landing).
# It is the public front door: https://japanfold.com and https://www.japanfold.com,
# with https://landing.japanfold.com kept alive as a 301 to the apex.
#
# Direct upload: only the built bytes are published, never the source. This repo is
# the single source of truth for the page — there is no mirror repo to keep in sync.
#
# Ships tt_bio/platform/landing/ exactly as committed on the canonical branch (git
# archive, so never a dirty tree), plus a _headers file that marks the versioned
# assets immutable and a small edge worker that folds landing.japanfold.com onto the
# apex.
#
# Usage:
#   scripts/deploy_landing.sh                      production -> japanfold.com
#   scripts/deploy_landing.sh --preview            preview -> a *.pages.dev URL, no traffic
#   scripts/deploy_landing.sh --allow-low-quality-hero    ship a hero below the bitrate floor
#
# Credentials: CLOUDFLARE_API_TOKEN (needs Pages:Edit) and CLOUDFLARE_ACCOUNT_ID,
# from the environment or ~/.coworker/cloudflare.env.
#
# One-time setup (already done): the japanfold-landing Pages project, with
# japanfold.com, www.japanfold.com and landing.japanfold.com all attached to it as
# custom domains. One project serves all three.
#
# Rollback — the landing:
#   1. Cloudflare dashboard > Workers & Pages > japanfold-landing > Deployments >
#      "Rollback to this deployment". Instant, and every past deploy is kept.
#   2. If Pages itself is the problem, point the hostnames back in DNS (zone
#      e89626607c673078e66e1f93315f946b). Tunnel origin is
#      e3d9384a-ade9-4198-bc17-ebc087bd7168.cfargotunnel.com, proxied.
#
# Rollback — the apex, back to the demo SPA on the Galaxy tunnel. PATCH both CNAMEs
# to the tunnel origin above (proxied, ttl 1):
#   japanfold.com      record a4248dae37574543cd208731175bdd2f
#   www.japanfold.com  record 9ebd50135dcef52355a3aecc75963de0
# api.japanfold.com (04b72999bf1657018fbb10cc1a357d72) is the published API base and
# is never touched by either phase, so the API keeps answering throughout.
set -euo pipefail

BRANCH=aiand-bio-platform
LANDING_DIR=tt_bio/platform/landing
PROJECT=japanfold-landing
WRANGLER=wrangler@4

PREVIEW=0
ALLOW_LOW_HERO=0
for arg in "$@"; do
    case "$arg" in
        --preview)                PREVIEW=1 ;;
        --allow-low-quality-hero) ALLOW_LOW_HERO=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain -- "$LANDING_DIR")" ]; then
    echo "refusing: $LANDING_DIR has uncommitted changes" >&2
    exit 1
fi
# Deploys ship reviewed bytes: the landing tree at HEAD must be identical to the one
# on the canonical branch. A worker branch may deploy (it often carries unrelated
# commits), but a landing change has to be merged and pushed first.
git fetch --quiet origin "$BRANCH"
if [ -n "$(git diff --name-only "origin/$BRANCH" HEAD -- "$LANDING_DIR")" ]; then
    echo "refusing: $LANDING_DIR differs from origin/$BRANCH — merge and push the landing change first" >&2
    git diff --stat "origin/$BRANCH" HEAD -- "$LANDING_DIR" >&2
    exit 1
fi
SHA=$(git rev-parse --short HEAD)

if [ -z "${CLOUDFLARE_API_TOKEN:-}" ] && [ -f "$HOME/.coworker/cloudflare.env" ]; then
    set -a; . "$HOME/.coworker/cloudflare.env"; set +a
fi
if [ -z "${CLOUDFLARE_API_TOKEN:-}" ] || [ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
    echo "refusing: CLOUDFLARE_API_TOKEN (Pages:Edit) and CLOUDFLARE_ACCOUNT_ID must be set" >&2
    exit 1
fi
export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Bundle = landing/ at HEAD (git archive: committed bytes only).
git archive HEAD "$LANDING_DIR" | tar -x -C "$TMP"
SITE="$TMP/site"
mv "$TMP/$LANDING_DIR" "$SITE"
# Asset filenames are versioned and never reused (a re-encode under an existing name
# serves the old cached bytes), so they can be cached forever. The HTML must not be:
# `/` is not matched by the `/*.html` rule and keeps Pages' default
# `max-age=0, must-revalidate`, which is what a redeployed page wants anyway.
cat > "$SITE/_headers" <<'EOF'
/assets/*
  Cache-Control: public, max-age=31536000, immutable

/*.html
  Cache-Control: public, max-age=300, must-revalidate
EOF

# The apex is the one canonical address. landing.japanfold.com stays attached to the
# project so old links keep working, and folds onto the apex.
#
# Only the hostname distinguishes the two cases, so it takes code at the edge. Both
# simpler options were tried against the live site and do not work: a _redirects line
# is matched on the path only and silently ignores a hostname in the source, and a
# functions/ directory is discovered relative to wrangler's working directory, not
# inside the uploaded bundle. _worker.js is, so that is what ships.
#
# A Cloudflare Redirect Rule would also do it, but needs a zone-scoped token this
# deploy does not have. _routes.json pins the worker to "/" so asset requests are
# served straight off the edge and never invoke it.
cat > "$SITE/_worker.js" <<'EOF'
const REDIRECT_FROM = "landing.japanfold.com";
const CANONICAL = "https://japanfold.com";

export default {
  fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname === REDIRECT_FROM) {
      return Response.redirect(CANONICAL + url.pathname + url.search, 301);
    }
    // The apex, www and the *.pages.dev preview URLs all serve the page directly.
    return env.ASSETS.fetch(request);
  },
};
EOF
cat > "$SITE/_routes.json" <<'EOF'
{"version": 1, "include": ["/"], "exclude": []}
EOF

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

# Hero quality floor: every <source> of the hero video must probe at or above the
# approved encode's bitrate per pixel. A "hosting efficiency" re-encode once cut the
# hero to a third of its bitrate and shipped silently; this gate makes a degraded hero
# a deliberate act — pass --allow-low-quality-hero to override. The floor is 1800 kbps
# at the 1440px desktop size and scales with pixel area, so the phone-sized variants
# are held to the same bits-per-pixel bar, not to a bitrate sized for four times their
# pixels.
if [ "$ALLOW_LOW_HERO" = 0 ]; then
python3 - "$SITE" <<'PY'
import json, re, subprocess, sys, pathlib
site = pathlib.Path(sys.argv[1])
html = (site / "index.html").read_text(encoding="utf-8")
m = re.search(r'<video class="hero-anim".*?</video>', html, re.S)
srcs = re.findall(r'<source src="(/[^"]+)"', m.group(0)) if m else []
FLOOR = 1_800_000
REF_AREA = 1440 * 1440
low = []
for s in srcs:
    f = site / s.lstrip("/")
    if not f.is_file():
        continue  # missing refs are the asset gate's job
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                          "format=bit_rate:stream=width,height",
                          "-of", "json", str(f)],
                         capture_output=True, text=True).stdout
    d = json.loads(out or "{}")
    br = int(d.get("format", {}).get("bit_rate") or 0)
    dims = [(st.get("width") or 0, st.get("height") or 0)
            for st in d.get("streams", []) if st.get("width")]
    floor = FLOOR * (dims[0][0] * dims[0][1]) / REF_AREA if dims else FLOOR
    if br < floor:
        low.append(f"{s} ({br / 1e3:.0f} kbps < {floor / 1e3:.0f} kbps floor)")
if low:
    sys.exit("refusing: hero source below the quality floor: "
             + ", ".join(low) + " — pass --allow-low-quality-hero to override")
print(f"hero quality gate: {len(srcs)} sources, all at or above the "
      f"1800 kbps @1440px floor (area-scaled)")
PY
fi

# Cloudflare Pages routes a deploy to production when its --branch is the project's
# production branch; any other branch is a preview on its own *.pages.dev URL.
if [ "$PREVIEW" = 1 ]; then
    TARGET="preview-$(git rev-parse --abbrev-ref HEAD | tr -c 'a-zA-Z0-9._-' '-')"
else
    TARGET="$BRANCH"
fi
npx --yes "$WRANGLER" pages deploy "$SITE" \
    --project-name "$PROJECT" \
    --branch "$TARGET" \
    --commit-hash "$(git rev-parse HEAD)" \
    --commit-message "landing from $BRANCH@$SHA" \
    --commit-dirty=true
echo "deployed $BRANCH@$SHA -> Cloudflare Pages $PROJECT (branch $TARGET)"
