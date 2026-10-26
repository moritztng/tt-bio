#!/usr/bin/env python3
"""bg_ab.py --arm NAME=TREE[:fast][@+lever+lever] ... --order a,b,b,a --out OUT.json : interleaved BoltzGen page A/B, one chip, seeded.

Each run is perf/bmk/design_cell.py's BoltzGen cell (`tt-bio design --model boltzgen --steps design` on
perf/dsfix/fixtures/bg_R3.yaml, one design = the wall between consecutive `batch k/N` stamps, the cold design and the
one paying the checkpoint switch dropped) with `--seed` added, so arms that compute the same thing write the same CIFs.
Run from anywhere with TT_VISIBLE_DEVICES set; each arm runs its own tree (PYTHONPATH and cwd = TREE). `@+a+b` runs the
arm under TT_BIO_LEVERS = that tree's BoltzGen set for the mode (LEVERS_NORMAL / LEVERS_FAST, WH_LEVERS_* with --wh) plus a and b.

OUT.json: runs[] = {arm, tree, fast, warm[], median, cif_sha256[], rc}; arms{} = pooled warm median per arm;
same_cifs{} = per arm pair whether every run's CIF list matches.
"""
import argparse, hashlib, json, os, re, statistics, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True, help="NAME=TREE[:fast][@+lever+lever]")
ap.add_argument("--order", required=True, help="comma-separated arm names, e.g. a,b,b,a")
ap.add_argument("--designs", type=int, default=6)
ap.add_argument("--wh", action="store_true", help="base an @ arm on the tree's Wormhole set (WH_LEVERS_*) where it has one")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timeout", type=int, default=1500)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

arms = {}
for spec in a.arm:
    name, rest = spec.split("=", 1)
    rest, _, extra = rest.partition("@")
    tree, _, flag = rest.partition(":")
    tree, fast, env = Path(tree).resolve(), flag == "fast", {}
    if extra:
        base = subprocess.run([sys.executable, "-c", "from tt_bio.boltzgen.model.models import boltz as b; "
                               f"m = '{'FAST' if fast else 'NORMAL'}'; print('+'.join(sorted(getattr(b, "
                               f"{'WH_' if a.wh else ''!r} + 'LEVERS_' + m, getattr(b, 'LEVERS_' + m)))))"],
                              cwd=tree, env=dict(os.environ, PYTHONPATH=str(tree)), capture_output=True, text=True,
                              check=True).stdout.strip().splitlines()[-1]
        env["TT_BIO_LEVERS"] = "+".join(sorted(set(base.split("+")) | {x for x in extra.split("+") if x}))
    arms[name] = (tree, fast, env)
res = {"seed": a.seed, "designs": a.designs, "card": os.environ.get("TT_VISIBLE_DEVICES"), "runs": []}
work = a.out.parent / "ab-work"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


for i, name in enumerate(a.order.split(",")):
    tree, fast, env = arms[name]
    out = work / f"{i}-{name}"
    subprocess.run(["rm", "-rf", str(out)])
    cmd = [sys.executable, "-u", "-m", "tt_bio.main", "design", "perf/dsfix/fixtures/bg_R3.yaml", "--model", "boltzgen",
           "--steps", "design", "--num_designs", str(a.designs), "--out_dir", str(out), "--seed", str(a.seed),
           "--debug", "--log"] + (["--fast"] if fast else [])
    run = {"arm": name, "tree": str(tree), "fast": fast, "levers": env.get("TT_BIO_LEVERS"), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    stamps, switched, cur, tail = {}, set(), 0, []
    p = subprocess.Popen(cmd, cwd=tree, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=dict(os.environ, PYTHONPATH=str(tree), PYTHONUNBUFFERED="1", **env))
    t_end = time.time() + a.timeout
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if m := re.search(r"batch (\d+)/(\d+)", line):
            cur = int(m.group(1)); stamps[cur] = time.time()
        elif "Switched checkpoint" in line:
            switched.add(cur + 1)
        if time.time() > t_end:
            p.terminate()
    run["rc"] = p.wait()
    walls = {k: stamps[k] - stamps[k - 1] for k in stamps if k - 1 in stamps}
    run["walls"] = {k: round(v, 3) for k, v in sorted(walls.items())}
    run["warm"] = [round(v, 3) for k, v in sorted(walls.items()) if k > 1 and k not in switched]
    run["median"] = round(statistics.median(run["warm"]), 3) if run["warm"] else None
    run["cif_sha256"] = [sha(c) for c in sorted((out / "intermediate_designs").glob("*.cif"))]
    if run["rc"] or len(run["cif_sha256"]) != a.designs:
        run["tail"] = "".join(tail)[-3000:]
    res["runs"].append(run)
    print(json.dumps({k: run[k] for k in ("arm", "rc", "median", "warm")}), flush=True)
    a.out.write_text(json.dumps(res, indent=1))

res["arms"] = {n: statistics.median(sum((r["warm"] for r in res["runs"] if r["arm"] == n), []) or [0]) for n in arms}
cifs = {n: [r["cif_sha256"] for r in res["runs"] if r["arm"] == n] for n in arms}
res["same_cifs"] = {f"{x}|{y}": all(c == cifs[x][0] for c in cifs[x] + cifs[y]) for x in arms for y in arms if x < y}
a.out.write_text(json.dumps(res, indent=1))
print(json.dumps({"arms": res["arms"], "same_cifs": res["same_cifs"]}))
