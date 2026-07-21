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
# The release asset is a single tarball named parity-fixtures-<tag>.tar.gz plus a
# parity-fixtures-<tag>.sha256 sidecar. The script downloads both, verifies the
# checksum, and extracts the tarball into --dest. It is idempotent: re-running
# over an existing tree only overwrites files the tarball provides.
#
# To create the release (maintainer, once per parity pass):
#   1. Harvest new fixtures:        python3 scripts/pharma_harvest_ref_fixtures.py ...
#   2. Tar the binary fixtures:     tar czf parity-fixtures-<tag>.tar.gz \
#                                      -C docs/implementation-parity-data ref-fixtures \
#                                      --include='*.cif' --include='*.a3m'  (or full tree)
#   3. sha256sum parity-fixtures-<tag>.tar.gz > parity-fixtures-<tag>.sha256
#   4. gh release create <tag> parity-fixtures-<tag>.tar.gz parity-fixtures-<tag>.sha256 \
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

ASSET="parity-fixtures-${TAG}.tar.gz"
SHA_ASSET="parity-fixtures-${TAG}.sha256"
API="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"

echo "Fetching parity fixtures: tag=${TAG} repo=${REPO} dest=${DEST}"

# Resolve the two asset download URLs from the release.
urls_json="$(curl -fsSL -H "Accept: application/vnd.github+json" "${API}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print("\n".join(a["browser_download_url"] for a in r.get("assets",[])))')"

tarball_url="$(printf '%s\n' "${urls_json}" | grep -F "/${ASSET}$" || true)"
sha_url="$(printf '%s\n' "${urls_json}" | grep -F "/${SHA_ASSET}$" || true)"
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
  ( cd "${tmp}" && sha256sum -c "${SHA_ASSET}" )
else
  echo "warn: no sha256 sidecar on release; skipping checksum verification." >&2
fi

mkdir -p "${DEST}"
tar xzf "${tmp}/${ASSET}" -C "$(dirname "${DEST}")"
echo "done: fixtures extracted under ${DEST}"
