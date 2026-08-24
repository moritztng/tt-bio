"""Has X-Cell shipped weights yet? One cheap check, meant to be run on a schedule.

X-Cell's architecture is ported and its performance measured, but no trained checkpoint exists, so
there is no accuracy claim to make. This is the trigger for making one. It compares the live
HuggingFace file list and the GitHub tree SHA against what was true when the port was written
(2026-08-24) and exits non-zero the moment either moves.

    python3 scripts/xcell_watch.py          # human-readable
    python3 scripts/xcell_watch.py --json   # for a cron/dashboard consumer

Exit codes: 0 nothing changed, 1 SOMETHING CHANGED (go look), 2 the check itself failed.
"""
import argparse
import json
import sys
import urllib.request

HF_API = "https://huggingface.co/api/models/Xaira-Therapeutics/X-Cell"
GH_API = "https://api.github.com/repos/Xaira-Therapeutics/X-Cell/git/trees/main?recursive=1"

# The baseline, verified 2026-08-24. Three files, none of them a checkpoint, and usedStorage equal
# to the overview PNG alone is the tell that no weight file is hiding behind a rename.
BASE_FILES = {".gitattributes", "README.md", "x-cell-overview.png"}
BASE_TREE_SHA = "7195c647b8316234ddaf51565701cfdaa939b443"
BASE_HF_SHA = "07737b494a3ea06c6a579d389fda0216ba61bf02"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".npz")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tt-bio-xcell-watch"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def check():
    out = {"changed": False, "notes": []}
    hf = _get(HF_API)
    files = {s["rfilename"] for s in hf.get("siblings", [])}
    out["hf_sha"] = hf.get("sha")
    out["hf_files"] = sorted(files)
    out["hf_weight_files"] = sorted(f for f in files if f.endswith(WEIGHT_SUFFIXES))
    out["downloads"] = hf.get("downloads")
    if out["hf_weight_files"]:
        out["changed"] = True
        out["notes"].append("WEIGHTS ON HUGGINGFACE: " + ", ".join(out["hf_weight_files"]))
    if files != BASE_FILES:
        out["changed"] = True
        added = sorted(files - BASE_FILES)
        removed = sorted(BASE_FILES - files)
        out["notes"].append(f"HF file list moved (added {added}, removed {removed})")
    if hf.get("sha") != BASE_HF_SHA:
        out["changed"] = True
        out["notes"].append(f"HF repo sha {hf.get('sha')} != baseline {BASE_HF_SHA}")

    gh = _get(GH_API)
    out["gh_tree_sha"] = gh.get("sha")
    if gh.get("sha") != BASE_TREE_SHA:
        out["changed"] = True
        out["notes"].append(f"GitHub tree sha {gh.get('sha')} != baseline {BASE_TREE_SHA}")
        # A populated src/xcell is the other half of the story: code without weights, or both.
        big = [t["path"] for t in gh.get("tree", [])
               if t["type"] == "blob" and t["path"].startswith("src/xcell/")
               and t.get("size", 0) > 6000]
        if big:
            out["notes"].append("src/xcell/ has grown real modules: " + ", ".join(big))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        r = check()
    except Exception as exc:
        print(f"xcell-watch: check FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(r, indent=2))
    elif r["changed"]:
        print("X-CELL UPSTREAM MOVED:")
        for n in r["notes"]:
            print("  " + n)
        print("\nNext: remap into tt_bio/xcell.py, run real-weight PCC on captured I/O, and only")
        print("then is there an accuracy claim. See state/xcell-bringup.md.")
    else:
        print(f"X-Cell unchanged: {len(r['hf_files'])} files on HF, no checkpoint, "
              f"tree {r['gh_tree_sha'][:12]}, {r['downloads']} downloads.")
    return 1 if r["changed"] else 0


if __name__ == "__main__":
    sys.exit(main())
