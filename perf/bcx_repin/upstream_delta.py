#!/usr/bin/env python3
"""What moved in upstream BindCraft 2 between the tree BCX pinned and the tree it must re-pin to.

Answers one question the campaign cannot afford to guess: does `perf/bcx_predictor/splice.py`
still apply after upstream's `fix/aa-bias` release? The splice replaces a whole `jax.lax.scan` by
patching `bindcraft.af.alphafold.model.layer_stack.layer_stack` and recognising closures by name,
so it breaks silently if the vendored AlphaFold moves under it.

Runs on any host with network and no Tenstorrent device. Takes about two seconds.

    python3 perf/bcx_repin/upstream_delta.py            # our pin -> upstream head
    python3 perf/bcx_repin/upstream_delta.py A B        # any two refs
"""
import json
import sys
import urllib.request

REPO = "PacesaLab/BindCraft2"
PINNED = "7a2dfdb"   # what qb1 and qb2 carry in /home/ttuser/bcx_e2e/bc2
HEAD = "301efdd"     # merge of PR #17 fix/aa-bias, 2026-09-24T17:16Z

# The splice's patch target and the two closures it recognises by name. If any of these move, the
# swap stops applying and the device arm silently runs BindCraft 2's own trunk instead of ours.
SPLICE_SURFACE = "bindcraft/af/"


def compare(base, head):
    url = f"https://api.github.com/repos/{REPO}/compare/{base}...{head}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    base, head = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else (PINNED, HEAD)
    d = compare(base, head)
    files = d["files"]
    print(f"{base}...{head}: {d['total_commits']} commits, {len(files)} files")

    touched = [f["filename"] for f in files if f["filename"].startswith(SPLICE_SURFACE)]
    print(f"\nvendored AlphaFold ({SPLICE_SURFACE}): {len(touched)} file(s) changed")
    for f in touched:
        print("  ", f)
    if not touched:
        print("   -> the splice's patch target and closure names are untouched; the swap still applies")

    print("\npython files changed outside the vendored AlphaFold:")
    for f in files:
        n = f["filename"]
        if n.endswith(".py") and not n.startswith(SPLICE_SURFACE):
            print(f"   {n:45s} +{f['additions']}/-{f['deletions']}")

    print("\nsettings and examples changed (these move the acceptance bar, not the port):")
    for f in files:
        n = f["filename"]
        if n.startswith(("settings/", "examples/")) and n.endswith(".json"):
            print(f"   {n:45s} +{f['additions']}/-{f['deletions']}")

    return 0 if not touched else 1


if __name__ == "__main__":
    sys.exit(main())
