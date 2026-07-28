"""AbAg-XM Tier-A acceptance gate: is every one of the 492 folds actually good?

Why this exists and `abag_xm_status*.py` is not enough. `progress.jsonl` is append-only by
design -- it is the evidence trail, and a fold that failed at 02:00 and succeeded at 06:00
leaves BOTH records forever. So a histogram over raw records can never reach
"0 fold_failed, 0 incomplete, 0 timed_out" no matter how healthy the slab is, and reporting
one as the acceptance number is either permanently red or quietly counting the wrong thing.

The acceptance question is per (target, model) pair, not per record:

  every pair has at least one `ok` record, and that record is defensible --
  50 CIFs, 50 per-sample PAEs, one engine tree, one config, stateable provenance.

A pair with no such record is outstanding, and its LAST record says why (that is the
histogram worth printing: what is still wrong, not what was ever wrong).

Usage:
    python3 scripts/abag_xm_acceptance.py [--host tt-quietbox --host tt-quietbox2]
                                          [--n_samples 50] [--mps 5] [--json out.json]

Exit status is the gate: 0 iff every pair is accepted.
"""
import argparse
import collections
import json
import pathlib
import socket
import subprocess
import sys

TARGETS = 164
MODELS = ("protenix-v2", "opendde-abag", "boltz2")
PROGRESS = "abag_xm/tier_a/progress.jsonl"
HOSTS = ("tt-quietbox", "tt-quietbox2")
MPS_SENSITIVE = {"boltz2"}   # the others ignore max_parallel_samples entirely


def _read(host):
    """Records from one host, local read if it is us, ssh otherwise. An unreachable host is
    fatal here (unlike in the status view): a partial slab cannot be accepted."""
    if host == socket.gethostname():
        p = pathlib.Path.home() / PROGRESS
        return p.read_text().splitlines() if p.exists() else []
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"ttuser@{host}",
         f"cat {PROGRESS}"], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"acceptance: cannot read {host} ({out.stderr.strip()[:200]}) -- "
                         f"a slab cannot be accepted from a subset of its hosts")
    return out.stdout.splitlines()


def _defects(r, n_samples, mps):
    """Everything that disqualifies an `ok` record from counting as an accepted fold.

    Note the PAE count: `abag_xm_generate.py` globs `<target>_model_*_pae.npz`, which excludes
    the extra backwards-compat `<target>_pae.npz` aggregate, so a healthy fold has exactly
    n_samples PAEs. Records showing n_samples+1 were not written by the harness at all -- they
    are post-hoc reconstructions by a looser-globbing tool, and they carry no provenance.
    """
    d = []
    if r.get("n_cifs") != n_samples:
        d.append(f"n_cifs={r.get('n_cifs')}")
    if r.get("n_paes") != n_samples:
        d.append(f"n_paes={r.get('n_paes')}")
    if r.get("n_samples") != n_samples:
        d.append(f"n_samples={r.get('n_samples')}")
    if r["model"] in MPS_SENSITIVE and r.get("mps") != mps:
        d.append(f"mps={r.get('mps')}")
    c, t = r.get("tt_bio_commit"), r.get("tt_bio_tree")
    if not t and (not c or c.endswith("-dirty")):
        # Pre-p6 records have no tt_bio_tree field; fall back to the commit, which at least
        # distinguishes "folded from a clean checkout" from "folded from a dirty one".
        d.append(f"provenance={c!r}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="append", default=None)
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--mps", type=int, default=5)
    ap.add_argument("--json", default=None, help="write the machine-readable verdict here")
    a = ap.parse_args()
    hosts = a.host or list(HOSTS)

    # last record per pair (why a pair is still outstanding) + the best ok record per pair
    last, accepted, rejected_ok = {}, {}, collections.defaultdict(list)
    n_records = collections.Counter()
    for h in hosts:
        for line in _read(h):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["target"], r["model"])
            except Exception:
                n_records["unparseable"] += 1
                continue
            n_records[h] += 1
            r["_host"] = h
            last[key] = r
            if r.get("status") != "ok":
                continue
            defects = _defects(r, a.n_samples, a.mps)
            if defects:
                rejected_ok[key].append((h, defects))
            else:
                accepted[key] = r

    pairs = {(t, m) for t in _targets(hosts) for m in MODELS}
    outstanding = sorted(pairs - set(accepted))

    print(f"records read     : " + "  ".join(f"{h}={n_records[h]}" for h in hosts)
          + (f"  unparseable={n_records['unparseable']}" if n_records["unparseable"] else ""))
    print(f"accepted folds   : {len(accepted)} / {len(pairs)}"
          f"   (ok, {a.n_samples} CIFs, {a.n_samples} PAEs, mps={a.mps}, provenance stateable)")
    print(f"OUTSTANDING      : {len(outstanding)}")

    # Engine + input homogeneity: the two things that make the slab one slab.
    trees = collections.Counter(r.get("tt_bio_tree") for r in accepted.values())
    print(f"\nengine trees     : {len(trees)} distinct over accepted folds")
    for t, n in trees.most_common():
        print(f"    {str(t)[:16]:<18} {n:>4} folds")
    msa = collections.Counter((r["target"], r.get("msa_sha")) for r in accepted.values())
    per_target = collections.defaultdict(set)
    for (t, s), _ in msa.items():
        per_target[t].add(s)
    split = {t: s for t, s in per_target.items() if len(s) > 1}
    print(f"MSA input hash   : {len(split)} target(s) folded against DIFFERENT MSA bytes"
          + ("" if not split else f"  {sorted(split)[:6]}"))

    if rejected_ok:
        print(f"\n`ok` records rejected on inspection: {len(rejected_ok)} pair(s)")
        hist = collections.Counter(d.split("=")[0] for v in rejected_ok.values()
                                   for _, ds in v for d in ds)
        for k, n in hist.most_common():
            print(f"    {k:<14} {n}")
        for (t, m), v in sorted(rejected_ok.items())[:10]:
            if (t, m) not in accepted:
                print(f"    {t} {m}: {v[0][1]}")

    if outstanding:
        print(f"\nwhy each outstanding pair is outstanding (its LAST record):")
        hist = collections.Counter()
        for key in outstanding:
            r = last.get(key)
            hist[r.get("status") if r else "never attempted"] += 1
        for k, n in hist.most_common():
            print(f"    {k:<18} {n}")
        print()
        for key in outstanding[:25]:
            r = last.get(key)
            if r is None:
                print(f"    {key[0]:<6} {key[1]:<14} never attempted")
                continue
            err = (r.get("stderr") or "").replace("\n", " ")
            print(f"    {key[0]:<6} {key[1]:<14} {r.get('status'):<14} "
                  f"{r.get('wall_s')}s {r['_host']}  {err[-110:]}")
        if len(outstanding) > 25:
            print(f"    ... and {len(outstanding) - 25} more")

    verdict = "ACCEPTED" if not outstanding else "NOT ACCEPTED"
    print(f"\nVERDICT: {verdict} -- {len(accepted)}/{len(pairs)} folds accepted, "
          f"{len(outstanding)} outstanding")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({
            "verdict": verdict, "accepted": len(accepted), "total": len(pairs),
            "outstanding": [list(k) for k in outstanding],
            "engine_trees": {str(k): v for k, v in trees.items()},
            "hosts": hosts, "n_samples": a.n_samples, "mps": a.mps}, indent=2))
    return 0 if not outstanding else 1


def _targets(hosts):
    """The manifest's 164 ids. Read from the local checkout; fall back to the fold YAMLs."""
    root = pathlib.Path(__file__).resolve().parent.parent
    try:
        import pandas as pd
        return pd.read_parquet(
            root / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"
        )["pdb_id"].tolist()
    except Exception:
        ids = sorted(p.stem for p in (root / "examples" / "abag_xm").glob("*.yaml"))
        if len(ids) != TARGETS:
            print(f"!! manifest unreadable and only {len(ids)} YAMLs on disk (expected "
                  f"{TARGETS}) -- the pair set below is incomplete", file=sys.stderr)
        return ids


if __name__ == "__main__":
    sys.exit(main())
