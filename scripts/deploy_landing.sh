#!/usr/bin/env bash
# Deploy the JapanFold landing page to Cloudflare Pages (project japanfold-landing,
# served as https://landing.japanfold.com).
#
# Direct upload: only the built bytes are published, never the source. This repo is
# the single source of truth for the page — there is no mirror repo to keep in sync.
#
# Ships tt_bio/platform/landing/ exactly as committed on the canonical branch (git
# archive, so never a dirty tree), plus a _headers file that marks the versioned
# assets immutable.
#
# Usage:
#   scripts/deploy_landing.sh                      production -> landing.japanfold.com
#   scripts/deploy_landing.sh --preview            preview -> a *.pages.dev URL, no traffic
#   scripts/deploy_landing.sh --allow-low-quality-hero    ship a hero below the bitrate floor
#
# Credentials: CLOUDFLARE_API_TOKEN (needs Pages:Edit) and CLOUDFLARE_ACCOUNT_ID,
# from the environment or ~/.coworker/cloudflare.env.
#
# One-time setup (already done): the japanfold-landing Pages project with production
# branch aiand-bio-platform, and landing.japanfold.com attached to it as a custom
# domain.
#
# Rollback, in order of preference:
#   1. Cloudflare dashboard > Workers & Pages > japanfold-landing > Deployments >
#      "Rollback to this deployment". Instant, and every past deploy is kept.
#   2. If Pages itself is the problem: point the landing.japanfold.com CNAME back at
#      e3d9384a-ade9-4198-bc17-ebc087bd7168.cfargotunnel.com (proxied). The platform
#      serves this same page through the tunnel, so recovery is one DNS call.
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
# serves the old cached bytes), so they can be cached forever. The HTML must not be.
cat > "$SITE/_headers" <<'EOF'
/assets/*
  Cache-Control: public, max-age=31536000, immutable

/*.html
  Cache-Control: public, max-age=300, must-revalidate
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
