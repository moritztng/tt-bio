import json, pathlib, shutil, sys

B = pathlib.Path("/home/cust-team/mthuening/p31")
MDIR = {"boltz2": "boltz2", "opendde-abag": "opendde",
        "protenix-v2": "protenix", "esmfold2": "esmfold2"}
LPFX = {"boltz2": "boltz2", "opendde-abag": "opendde",
        "protenix-v2": "protenix", "esmfold2": "esmfold2"}

ok = set()
for line in (B / "results.jsonl").read_text().splitlines():
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("rung") == 512 and r.get("rc") == 0:
        ok.add((r["model"], r["target"], r.get("chunk")))

tasks = (B / "tasks.txt").read_text().splitlines()
n_claim = n_dir = n_log = 0
for i, line in enumerate(tasks, 1):
    m, t, rung, seed, c, k = line.split()
    if (m, t, int(c)) in ok:
        continue
    claim = B / "claims" / str(i)
    if claim.exists():
        shutil.rmtree(claim)
        n_claim += 1
    d = B / MDIR[m] / (t + "_c" + c)
    if d.exists():
        shutil.rmtree(d)
        n_dir += 1
    for log in B.glob(LPFX[m] + "_" + t + "_c" + c + "*.log"):
        log.unlink()
        n_log += 1
print("purged: %d claims, %d outdirs, %d logs" % (n_claim, n_dir, n_log))
print("ok cells:", len(ok), "of", len(tasks))
