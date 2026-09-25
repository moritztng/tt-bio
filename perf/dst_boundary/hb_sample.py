"""Record every lease renewal of the jobs one owner holds on this host's controller.

lease_until is set to (heartbeat time + lease) by each renewal, so its distinct values
are the renewal times exactly; the sampler only has to see each one. Read-only.
Usage: python3 hb_sample.py <owner> <out.jsonl> [max_s]
"""
import glob, json, os, sqlite3, sys, time

owner, out, max_s = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2400
db = max(glob.glob("/tmp/tt-bio-controller-*/controller.sqlite3"), key=os.path.getmtime)
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
seen, t0, active = set(), time.time(), False
with open(out, "a") as f:
    while time.time() - t0 < max_s:
        rows = c.execute("SELECT j.run_id, j.job_id, j.name, j.worker_id, j.lease_until, j.status "
                         "FROM jobs j JOIN runs r USING(run_id) WHERE r.owner=? AND j.status='running'",
                         (owner,)).fetchall()
        for run, job, name, worker, lease_until, _ in rows:
            if (job, lease_until) not in seen:
                seen.add((job, lease_until))
                f.write(json.dumps({"t": time.time(), "run": run, "job": job, "name": name,
                                    "worker": worker, "lease_until": lease_until}) + "\n")
                f.flush()
        active = active or bool(rows)
        if active and not rows:
            break
        time.sleep(0.5)
