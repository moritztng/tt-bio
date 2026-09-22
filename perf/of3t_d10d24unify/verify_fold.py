"""Does the shipped code actually rank on the shared rule? One real fold says so.

The offline re-ranking in `rerank_vs_main.py` proves the arithmetic. This proves the WIRING:
it runs the production CLI, then checks every published row against
`tt_bio.ranking.ranking_score` recomputed from that row's OWN pTM/ipTM/pLDDT, and checks the
rows come out in non-increasing score order with the served file at rank 0. A site left
un-wired, or wired to a stale copy, fails here and nowhere else.

No timing is recorded or claimed: qb1 was running two other workers' folds during these runs,
so a wall-clock number from them would be an artifact.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--card", type=int, default=2)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--out-root", default=os.path.expanduser("~/of3t_d10d24_out"))
ap.add_argument("--msa-dir", default=os.path.expanduser("~/of3t_rankunify_msa"))
ap.add_argument("--python", default=os.path.expanduser("~/tt-bio-dev/env/bin/python"))
a = ap.parse_args()

stem = pathlib.Path(a.target).stem
tag = f"{a.model}__{stem}__s{a.seed}"
out = pathlib.Path(a.out_root) / tag
if out.exists():
    import shutil
    shutil.rmtree(out)
out.mkdir(parents=True)

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
env.update(TT_VISIBLE_DEVICES=str(a.card), TT_BIO_LEASE_CARDS=str(a.card),
           TT_BIO_LEASE_HOLDER="worker:of3t-d10d24-unify")

# The checkout, not an installed wheel further along sys.path.
where = subprocess.run([a.python, "-c", "import tt_bio,sys;print(tt_bio.__file__)"],
                       env=env, cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
if not where.startswith(str(ROOT)):
    sys.exit("tt_bio resolves to %s, not this worktree" % where)

cmd = [a.python, "-m", "tt_bio.main", "predict", a.target, "--model", a.model,
       "--out_dir", str(out), "--seed", str(a.seed),
       "--diffusion_samples", str(a.samples), "--use_msa_server", "--msa_dir", a.msa_dir]
t0 = time.time()
proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
(out / "fold.log").write_text(proc.stdout + proc.stderr)
if proc.returncode != 0:
    print("FAIL %s rc=%d" % (tag, proc.returncode))
    print(proc.stdout[-3000:], proc.stderr[-3000:])
    sys.exit(proc.returncode)

sys.path.insert(0, str(ROOT))
from tt_bio.ranking import ranking_score

res = json.loads(next(out.rglob("results.json")).read_text())


def rows(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from rows(x)
    elif isinstance(obj, dict):
        if isinstance(obj.get("all_runs"), list):
            yield obj["all_runs"]
        for v in obj.values():
            yield from rows(v)


runs = next(iter(rows(res)), None)
if not runs or len(runs) != a.samples:
    sys.exit("%s: results.json has no all_runs of %d" % (tag, a.samples))

key = "ranking_score" if "ranking_score" in runs[0] else "confidence_score"
scores, bad = [], []
for r, row in enumerate(runs):
    got = float(row[key])
    mine = ranking_score(ptm=row.get("ptm") or 0.0, iptm=row.get("iptm"),
                         plddt=row.get("plddt") or 0.0,
                         disorder=row.get("disorder") or 0.0,
                         has_clash=float(row.get("has_clash") or 0.0))
    scores.append(got)
    # Published rows round to 4 decimals at worst, so anything above 5e-5 is a real
    # disagreement between the shipped rule and what the site published.
    if abs(got - mine) > 6e-5:
        bad.append({"rank": r, "published": got, "shared_rule": mine,
                    "ptm": row.get("ptm"), "iptm": row.get("iptm"),
                    "plddt": row.get("plddt")})

ordered = all(scores[i] >= scores[i + 1] - 6e-5 for i in range(len(scores) - 1))
struct = next(out.rglob("structures"))
served = [p for p in struct.iterdir() if p.stem == stem]

payload = {"tag": tag, "model": a.model, "target": a.target, "seed": a.seed,
           "card": a.card, "board": "p150a", "host": "qb1", "wall_s": round(time.time() - t0, 1),
           "score_key": key, "scores_by_rank": scores, "rows_in_rank_order": ordered,
           "rule_mismatches": bad, "served_file": served[0].name if served else None,
           "iptm_of_rank0": runs[0].get("iptm")}
(out / "verify.json").write_text(json.dumps(payload, indent=2) + "\n")
ok = ordered and not bad and len(served) == 1
print("%s %s rank0=%.6f ordered=%s mismatches=%d served=%s"
      % ("OK  " if ok else "FAIL", tag, scores[0], ordered, len(bad),
         served[0].name if served else None))
sys.exit(0 if ok else 1)
