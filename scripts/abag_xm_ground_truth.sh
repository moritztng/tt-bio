#!/usr/bin/env bash
# abag_xm_ground_truth.sh -- get the AbAg-XM ground-truth structures onto this host.
#
# 170 experimental mmCIF files, 145 MB. They are append-only reference data, so they ship as a
# GitHub Release asset rather than as git blobs: committed directly they were 141 files and
# 1,897,842 added lines on the branch, which is what the binary-fixtures policy exists to stop.
# Every consumer (abag_xm_generate.py, abag_xm_labels_campaign.py) prefers $GT_DIR and falls back
# to whatever the checkout carries, so this script is the one thing a fresh clone needs.
#
#   fetch    download + verify into $GT_DIR (default ~/abag_xm/ground_truth)
#   verify   check what is already there against the manifest
#   stage    build the tarball + manifest from $GT_DIR, ready to upload as an asset
#
# `stage` is deliberately separate from uploading. Publishing to a public repo is Moritz's call,
# and the expensive, error-prone half is building a reproducible archive -- so do that here and
# leave `gh release upload` as a one-liner someone runs knowingly.
set -u
GT_DIR="${ABAG_XM_GT_DIR:-$HOME/abag_xm/ground_truth}"
STAGE_DIR="${ABAG_XM_STAGE_DIR:-$HOME/abag_xm/release_staging}"
TARBALL=abag-xm-ground-truth.tar.gz
MANIFEST=abag-xm-ground-truth.sha256
REPO="${ABAG_XM_REPO:-moritztng/tt-bio}"
TAG="${ABAG_XM_GT_TAG:-abag-xm-ground-truth-v1}"
URL="https://github.com/$REPO/releases/download/$TAG"
say(){ echo "[ground-truth] $*"; }

# Sorted, name-only checksums: a manifest that embeds absolute paths or directory order cannot be
# compared between two hosts, which is the whole point of having one.
manifest_of(){ (cd "$1" && find . -maxdepth 1 -name '*.cif' -printf '%P\n' | LC_ALL=C sort \
                 | xargs -r sha256sum); }

case "${1:-verify}" in
  stage)
    [ -d "$GT_DIR" ] || { say "no $GT_DIR to stage from"; exit 1; }
    n=$(find "$GT_DIR" -maxdepth 1 -name '*.cif' | wc -l)
    [ "$n" -gt 0 ] || { say "$GT_DIR has no .cif files"; exit 1; }
    mkdir -p "$STAGE_DIR"
    manifest_of "$GT_DIR" > "$STAGE_DIR/$MANIFEST"
    # --sort=name and a fixed mtime/owner so two runs over the same inputs produce the same
    # bytes; without them the archive hash changes every time and cannot be checked.
    tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-01-01 00:00:00Z' \
        -czf "$STAGE_DIR/$TARBALL" -C "$GT_DIR" $(cd "$GT_DIR" && ls *.cif | LC_ALL=C sort)
    say "staged $n CIFs -> $STAGE_DIR/$TARBALL ($(du -h "$STAGE_DIR/$TARBALL" | cut -f1))"
    say "archive sha256: $(sha256sum "$STAGE_DIR/$TARBALL" | cut -d' ' -f1)"
    say "upload with:  gh release create $TAG --repo $REPO --title 'AbAg-XM ground-truth structures' \\"
    say "                $STAGE_DIR/$TARBALL $STAGE_DIR/$MANIFEST" ;;
  fetch)
    mkdir -p "$GT_DIR"
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    say "fetching $URL/$TARBALL"
    curl -fSL --retry 3 -o "$tmp/$TARBALL" "$URL/$TARBALL" || {
      say "download failed. If the asset does not exist yet, run \`$0 stage\` on a host that has"
      say "  the files and upload it -- see the gh command that prints."; exit 1; }
    curl -fSL --retry 3 -o "$tmp/$MANIFEST" "$URL/$MANIFEST" || { say "manifest download failed"; exit 1; }
    tar -xzf "$tmp/$TARBALL" -C "$GT_DIR"
    (cd "$GT_DIR" && sha256sum -c "$tmp/$MANIFEST" --quiet) \
      && say "fetched and verified $(grep -c . "$tmp/$MANIFEST") CIFs into $GT_DIR" \
      || { say "CHECKSUM MISMATCH after extract -- do not use $GT_DIR"; exit 1; } ;;
  verify)
    [ -d "$GT_DIR" ] || { say "$GT_DIR absent -- run \`$0 fetch\`"; exit 1; }
    n=$(find "$GT_DIR" -maxdepth 1 -name '*.cif' | wc -l)
    say "$GT_DIR: $n CIFs, $(du -sh "$GT_DIR" | cut -f1)"
    if [ -f "$STAGE_DIR/$MANIFEST" ]; then
      (cd "$GT_DIR" && sha256sum -c "$STAGE_DIR/$MANIFEST" --quiet) \
        && say "all files match $STAGE_DIR/$MANIFEST" || { say "MISMATCH vs staged manifest"; exit 1; }
    else
      say "no staged manifest to compare against (run \`$0 stage\` to make one)"
    fi ;;
  *) echo "usage: $0 {fetch|verify|stage}" >&2; exit 2 ;;
esac
