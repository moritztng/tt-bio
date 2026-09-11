"""Run one `tt-bio predict` fold with the mask-after-move flag forced on or off, and report
what the trimul actually did inside it.

A fold-level sha comparison is worth nothing unless E6 ran, and E6 only reaches a trimul past
TRIANGLE_MULT_L1_MAX_SEQ = 352 (below that the trimul is on the L1 path and both E6 gates want a
DRAM output config). So this prints `reblock_permute_gated`'s own call counter at exit alongside
the CIF sha256, and a fold that reports 0 gated moves in the flag-on arm scores nothing.

    python3 perf/b2x_crossmodel/fold_arm.py --flag 1 --tag B -- \
        predict examples/615.yaml --model openfold3 --single_sequence ...
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", required=True, help="where to write the arm's record")
    ap.add_argument("--results", required=True, help="results.json the fold will write")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    os.environ["TT_BIO_TRIMUL_MASK_AFTER_MOVE"] = str(a.flag)
    from tt_bio import reblock_permute as RB
    from tt_bio import tenstorrent as T
    assert T._TRIMUL_MASK_AFTER_MOVE == bool(a.flag), "the env flag did not reach the module"

    argv = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    from tt_bio.main import cli
    rc = 0
    try:
        cli(argv, standalone_mode=False)
    except SystemExit as e:
        rc = e.code or 0

    rec = {"tag": a.tag, "flag": bool(a.flag), "rc": rc,
           "e6_moves": RB.STATS_GATED[0],
           "plain_moves": RB.STATS[0], "plain_rejects": RB.STATS[1],
           "rejects": {f"{k[0]}{list(k[1])}": v for k, v in RB.REJECTS.items()}}
    # The results dir is named after the ENGINE, not the --model string (protenix-v2 writes
    # protenix_results_615), so accept a root and find the one results.json under it.
    results = a.results
    if not os.path.isfile(results):
        root = results
        while root and not os.path.isdir(root):
            root = os.path.dirname(root)
        hits = sorted(glob.glob(os.path.join(root, "**", "results.json"), recursive=True))
        assert hits, f"no results.json under {root}"
        results = hits[0]
    a.results = results
    res = json.load(open(results))
    rec["results"] = res
    root = os.path.dirname(a.results)
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if n.endswith((".cif", ".pdb")):
                p = os.path.join(dirpath, n)
                rec.setdefault("structures", {})[os.path.relpath(p, root)] = \
                    hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(rec, open(a.report, "w"), indent=2)
    print(f"ARM {a.tag} flag={a.flag} e6_moves={rec['e6_moves']} "
          f"structures={rec.get('structures')}", flush=True)


if __name__ == "__main__":
    main()
