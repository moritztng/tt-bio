#!/usr/bin/env bash
# Fetch externalized parity reference fixtures from a GitHub Release.
#
# The large binary fixtures (CIF structures, A3M MSAs) that used to live in
# docs/implementation-parity-data/ref-fixtures/ are externalized to GitHub Release
# assets to keep the repo small. The small provenance JSONs/yaml/csv stay
# committed (they are the evidence a reader checks); this script restores the
# binaries so a fresh checkout can reproduce a parity leg end-to-end.
#
# Usage:
#   scripts/fetch_parity_fixtures.sh [--tag <tag>] [--repo <owner/repo>] [--dest <dir>]
#
# Defaults:
#   --tag   parity-fixtures-latest   (release tag carrying the fixture tarball)
#   --repo  moritztng/tt-bio         (the GitHub repo hosting the release)
#   --dest  docs/implementation-parity-data/ref-fixtures  (extract root)
#
# The release asset is a single tarball named <tag>.tar.gz plus a
# <tag>.sha256 sidecar. The script downloads both, verifies the
# checksum, and extracts the tarball into --dest. It is idempotent: re-running
# over an existing tree only overwrites files the tarball provides.
#
# To create the release (maintainer, once per parity pass):
#   1. Harvest new fixtures:        python3 scripts/pharma_harvest_ref_fixtures.py ...
#   2. Tar the binary fixtures:     tar czf <tag>.tar.gz \
#                                      -C docs/implementation-parity-data ref-fixtures \
#                                      --include='*.cif' --include='*.a3m' \
#                                      --include='*.npz'  (or full tree)
#   3. sha256sum <tag>.tar.gz > <tag>.sha256
#   4. gh release create <tag> <tag>.tar.gz <tag>.sha256 \
#         --repo moritztng/tt-bio --notes "Externalized parity reference fixtures"
#   5. Commit the new provenance JSONs (meta.json/results.json) with the asset tag recorded.
#
set -euo pipefail

TAG="parity-fixtures-latest"
REPO="moritztng/tt-bio"
DEST="docs/implementation-parity-data/ref-fixtures"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)  TAG="$2";  shift 2;;
    --repo) REPO="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0;;
    *)
      echo "error: unknown argument: $1" >&2; exit 2;;
  esac
done

ASSET="${TAG}.tar.gz"
SHA_ASSET="${TAG}.sha256"
API="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"

echo "Fetching parity fixtures: tag=${TAG} repo=${REPO} dest=${DEST}"

# Resolve the two asset download URLs from the release.
release_json="$(curl -fsSL -H "Accept: application/vnd.github+json" "${API}")"
urls_json="$(printf '%s' "${release_json}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print("\n".join(a["browser_download_url"] for a in r.get("assets",[])))')"
# Asset id for the tarball, used as the CDN-free fallback below.
asset_id="$(printf '%s' "${release_json}" | python3 -c '
import json, sys
r = json.load(sys.stdin)
print(next((str(a["id"]) for a in r.get("assets", []) if a["name"] == sys.argv[1]), ""))
' "${ASSET}")"

# Match the asset URL by its path suffix. grep -F (fixed-string) is used so the
# asset name is matched literally (no regex escaping of dots); the leading "/"
# binds the match to the path segment so one asset name can't substring-match
# another (e.g. .tar.gz vs .tar.gz.bak). Do NOT append a "$" anchor under -F:
# fixed-string mode treats "$" as a literal, not end-of-line.
tarball_url="$(printf '%s\n' "${urls_json}" | grep -F "/${ASSET}" || true)"
sha_url="$(printf '%s\n' "${urls_json}" | grep -F "/${SHA_ASSET}" || true)"
if [[ -z "${tarball_url}" ]]; then
  echo "error: release ${TAG} on ${REPO} has no asset named ${ASSET}." >&2
  echo "       Create it (see scripts/fetch_parity_fixtures.sh header) and retry." >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
curl -fsSL -o "${tmp}/${ASSET}" "${tarball_url}"

if [[ -n "${sha_url}" ]]; then
  curl -fsSL -o "${tmp}/${SHA_ASSET}" "${sha_url}"
  # Compare the hash FIELD, not `sha256sum -c`. The sidecar records whatever path
  # the maintainer hashed (the published one records an absolute /tmp path), and
  # `-c` re-opens that exact path -- which fails everywhere except the machine that
  # generated it. The checksum is about the bytes, so verify the bytes.
  want="$(head -n1 "${tmp}/${SHA_ASSET}")"; want="${want%% *}"
  got="$(sha256sum "${tmp}/${ASSET}")"; got="${got%% *}"
  if [[ "${want}" != "${got}" && -n "${asset_id}" ]]; then
    # browser_download_url is served through a CDN that can hold a stale copy for
    # minutes after the asset is re-uploaded, while the 96-byte sha256 sidecar
    # propagates at once. That reads as a checksum mismatch even though nothing is
    # corrupt. The API asset endpoint serves the stored bytes directly, so retry
    # there once before calling it a failure.
    echo "checksum mismatch on the CDN copy; retrying via the release asset API" >&2
    curl -fsSL -H "Accept: application/octet-stream" -o "${tmp}/${ASSET}" \
      "https://api.github.com/repos/${REPO}/releases/assets/${asset_id}"
    got="$(sha256sum "${tmp}/${ASSET}")"; got="${got%% *}"
  fi
  if [[ "${want}" != "${got}" ]]; then
    echo "error: checksum mismatch for ${ASSET}: expected ${want}, got ${got}" >&2
    exit 1
  fi
  echo "checksum OK (${got})"
else
  echo "warn: no sha256 sidecar on release; skipping checksum verification." >&2
fi

mkdir -p "${DEST}"
tar xzf "${tmp}/${ASSET}" -C "$(dirname "${DEST}")"
echo "done: fixtures extracted under ${DEST}"
