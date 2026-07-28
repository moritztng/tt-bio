"""QC sweep over produced label JSONs: which metrics actually carry a value, and why not.

Each label sub-script returns its own dict, and abag_xm_labels._run stores {"_error": ...} in
place of that dict when the script fails. A per-sample record therefore always has all four
metric keys present whether or not the metric was computed, so "the key is there" says nothing --
you have to look inside. This counts real values vs errors per metric and groups the error
messages, so a systematic failure is visible while the slab is still being built.

Usage: python3 label_qc.py <labels_dir> [<labels_dir> ...]
"""
import collections
import json
import pathlib
import sys

# metric -> the scalar key inside its dict that carries the actual value
SCALAR = {
    "dockq": "dockq",
    "epitope_jaccard": "epitope_jaccard",
    "interface_lddt": "interface_lddt",
    "cdr_rmsd": None,   # nested under "cdrs"; handled specially
}


def cdr_value(sub):
    """cdr_rmsd nests per-CDR values under "cdrs"; treat it as computed if any CDR has a number."""
    cdrs = sub.get("cdrs")
    if isinstance(cdrs, dict):
        for v in cdrs.values():
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (int, float)):
                        return vv
    return None


def main():
    ok = collections.Counter()
    err = collections.Counter()
    msgs = collections.defaultdict(collections.Counter)
    per_target_err = collections.defaultdict(set)
    nfiles = nsamples = 0

    for d in sys.argv[1:]:
        for f in sorted(pathlib.Path(d).glob("*.json")):
            try:
                doc = json.loads(f.read_text())
            except Exception:
                print(f"!! unparseable: {f}")
                continue
            nfiles += 1
            for rec in doc.get("samples", []):
                nsamples += 1
                for m in SCALAR:
                    sub = rec.get(m)
                    if not isinstance(sub, dict):
                        err[m] += 1
                        msgs[m]["<missing or non-dict>"] += 1
                        per_target_err[m].add(doc.get("target"))
                        continue
                    if "_error" in sub:
                        err[m] += 1
                        msgs[m][str(sub["_error"])[:110]] += 1
                        per_target_err[m].add(doc.get("target"))
                        continue
                    val = cdr_value(sub) if m == "cdr_rmsd" else sub.get(SCALAR[m])
                    if isinstance(val, (int, float)):
                        ok[m] += 1
                    else:
                        err[m] += 1
                        msgs[m]["<no numeric value in dict>"] += 1
                        per_target_err[m].add(doc.get("target"))

    print(f"label files: {nfiles}   per-sample records: {nsamples}\n")
    for m in SCALAR:
        tot = ok[m] + err[m]
        pct = 100.0 * ok[m] / tot if tot else 0.0
        print(f"{m:<16} computed {ok[m]:5d}/{tot:5d}  ({pct:5.1f}%)   "
              f"affected targets: {len(per_target_err[m])}")
        for msg, n in msgs[m].most_common(4):
            print(f"      {n:5d}x  {msg}")
    print()


if __name__ == "__main__":
    main()
