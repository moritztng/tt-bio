#!/usr/bin/env python3
"""Rebuild, in THIS host's claim queue, the campaign jobs that live only on an unreachable peer.

Why this exists: the two hosts share no filesystem, so `plan_queue` split the Q2 jobs into
one queue per host on that host's own disk. When qb1 went down mid-campaign (state doc p31)
its unclaimed queue became unreachable, and because the split is capacity-weighted LPT it had
put *one chunk of a two-chunk target* on each host for many targets. A chunked target is only
quotable at the full N=1000, so every one of those targets is blocked on a peer that may never
come back -- 8 protenix targets and 5 boltz2 jobs, not the single target the earlier
in-flight-inventory read suggested.

What counts as already covered on this host (never re-folded):
  - the out_dir holds labels.json (the fold ran here, or its labels were gathered from the peer),
  - or the out_dir holds a complete fold (results.json ok + n_samples CIFs),
  - or a queue file for that (model, target, chunk) already exists here, claimed or not.

Everything else is missing and gets written. Naming keeps the existing convention so the
drivers' longest-first `sorted(glob("job_*.json"))` order stays meaningful:
  - a rescue whose SIBLING CHUNK is already queued here reuses the sibling's rank, so the pair
    is claimed back-to-back and the target becomes quotable within about one fold of its partner
    instead of a whole queue later;
  - anything else trails the queue at rank >= TRAIL_RANK;
  - `--first` pins one job at rank 000 (used for the last opendde chunk, which alone blocks the
    primary deliverable).

Whether a target was planned as one 1000-sample fold or two 500-sample chunks CANNOT be
recomputed: `jobs_for_model` applies the ">4h projected" rule to `measured_walls()`, which grows
as the campaign's own progress records land, so a target planned chunked at 08:00 can compute as
unchunked by 17:00 (protenix 9gei did exactly this -- its opendde wall arrived after the queue
was written). Recomputing it would fold 1000 fresh samples into `9gei/` while `9gei_c1/` already
holds 500 from the same plan. So the existing queue and the already-folded dirs are taken as the
PLAN OF RECORD for that decision, and a rescue is derived from the sibling chunk's own JSON
(flip chunk, seed, out_dir). Only a target with no trace at all on this host is recomputed, and
then its arity must agree with the evidence-bearing targets or the script refuses to apply.

For the same reason the recomputation is pinned to the walls as they stood when the queue was
written (`--as_of`, default the queue's own mtime) rather than as they stand now. With the
snapshot pinned the recomputation reproduces every queued job exactly, which is what earns the
right to use it for the targets that have no local trace at all.

Pass the same `--scales` `plan_queue` was called with: the per-model cost ratios are what put a
target on its side of the 4h threshold.
"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abag_xm_saturation_run import (BASE, MODELS, PROJ, jobs_for_model, complete)

TRAIL_RANK = 100
# 9q6z folds a byte-identical input to 9q6y (state doc p5: the duplicate is in the dataset, not
# in our YAMLs), so its Q2 folds were deliberately held back in q2/held_duplicate_9q6z/. A
# rescue must not resurrect them.
HELD = {("protenix-v2", "9q6z"), ("boltz2", "9q6z")}


def key(job):
    return job["model"], job["target"], job["chunk"]


def queue_jobs(qdir):
    """(model, target, chunk) -> (queue path, job dict) for every job file in this queue."""
    out = {}
    for jf in sorted(qdir.glob("job_*.json")):
        job = json.loads(jf.read_text())
        out[key(job)] = (jf, job)
    return out


def covered(job):
    """True if this host already holds the fold or its gathered labels."""
    return (Path(job["out_dir"]) / "labels.json").exists() or complete(job)


def out_dir(model, target, chunk):
    return BASE / MODELS[model][1] / (target if chunk is None else f"{target}_c{chunk}")


def planned_arity(model, target, have):
    """CHUNKED / WHOLE for this (model, target) from the plan of record on this host.

    Evidence is a queue file or an already-folded out_dir; None means this host has no trace
    of the target at all (its jobs went to the peer) and the arity has to be inferred.
    """
    chunks = {k[2] for k in have if k[0] == model and k[1] == target}
    chunks |= {j for j in (None, 0, 1)
               if (out_dir(model, target, j) / "labels.json").exists()}
    if not chunks:
        return None
    return "WHOLE" if chunks == {None} else "CHUNKED"


def check_two_chunk_assumption(qdir, models):
    """Fail loudly if any target is split into more than two chunks.

    Three places here assume a chunked target has exactly chunks {0, 1}: the sibling lookup
    (`1 - chunk`), the arity scan over (None, 0, 1), and the emit loop over (0, 1). That holds
    for the plan as frozen -- `jobs_for_model` only ever emits 2 -- but if the boltz2 arm is
    ever re-chunked for host memory (the OOM finding), a chunk 2 would make the sibling lookup
    ask for chunk -1 and silently rescue a job with the WRONG seed and out_dir, i.e. quietly
    corrupt a target's sample set rather than fail. An assumption that would fail silently is
    worth an explicit check even while it still holds.
    """
    bad = set()
    for jf in qdir.glob("job_*.json"):
        c = json.loads(jf.read_text()).get("chunk")
        if c not in (None, 0, 1):
            bad.add(f"{jf.name} (chunk {c})")
    for model in models:
        parent = BASE / MODELS[model][1]
        for d in parent.glob("*_c*"):
            m = re.fullmatch(r".+_c(\d+)", d.name)
            if m and int(m.group(1)) > 1:
                bad.add(f"{model}/{d.name}")
    if bad:
        raise SystemExit(
            "this script assumes at most two chunks per target and would silently derive a "
            "wrong sibling seed otherwise; generalise sibling_rescue/planned_arity first. "
            "Found: " + ", ".join(sorted(bad)))


def sibling_rescue(sib_path, sib_job, chunk):
    """The other chunk of a queued job, derived from it so both chunks share one plan."""
    job = dict(sib_job)
    base_seed = MODELS[job["model"]][2]
    job["chunk"] = chunk
    job["seed"] = base_seed + 1000 * chunk
    job["out_dir"] = str(BASE / MODELS[job["model"]][1] / f"{job['target']}_c{chunk}")
    # keep the sibling's rank so the pair is claimed together
    rank = sib_path.name.split("_")[1]
    return job, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="this host's claim-queue dir")
    ap.add_argument("--models", default="opendde-abag,protenix-v2,boltz2")
    # Measured per-model cost ratio vs opendde (state doc p5), i.e. the scales plan_queue ran
    # with. They decide chunking, so a rescue must reuse them.
    ap.add_argument("--scales", default="protenix-v2=1.34,boltz2=0.68",
                    help="MODEL=FACTOR,... as passed to plan_queue")
    ap.add_argument("--as_of", type=float, default=None, metavar="EPOCH",
                    help="pin measured walls to this time (default: the queue's own mtime)")
    ap.add_argument("--first", default=None, metavar="MODEL/TARGET[_cJ]",
                    help="pin this job at rank 000 so the next free card takes it")
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    a = ap.parse_args()

    qdir = Path(a.queue)
    if not qdir.is_dir():
        raise SystemExit(f"no such queue dir: {qdir}")
    models = a.models.split(",")
    check_two_chunk_assumption(qdir, models)
    have = queue_jobs(qdir)

    scales = {}
    for part in filter(None, a.scales.split(",")):
        m, _, f = part.partition("=")
        scales[m] = float(f)
    as_of = a.as_of
    if as_of is None:
        mtimes = [jf.stat().st_mtime for jf, _ in have.values()]
        if not mtimes:
            raise SystemExit(f"{qdir} holds no job files: cannot date the plan (pass --as_of)")
        # OLDEST, i.e. when plan_queue wrote the batch. Rescues land later, so taking the
        # newest would date the plan to the last rescue and re-introduce the drift this
        # snapshot exists to remove.
        as_of = min(mtimes)
    print(f"walls pinned to {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(as_of))} "
          f"({len(have)} jobs in the queue)")
    computed = {}
    for m in models:
        for job in jobs_for_model(m, sorted(PROJ), scales.get(m, 1.0), as_of):
            computed[key(job)] = job

    missing, trail, drift = [], TRAIL_RANK, []
    for model in models:
        for target in sorted(PROJ):
            if (model, target) in HELD:
                continue
            recomputed_arity = "CHUNKED" if (model, target, 0) in computed else "WHOLE"
            arity = planned_arity(model, target, have)
            inferred = arity is None
            if inferred:  # no trace of this target here: the recomputation is the only guide
                arity = recomputed_arity
            elif arity != recomputed_arity:
                # The plan of record wins; recorded because it discredits any INFERRED arity.
                drift.append(f"{model}/{target}: plan says {arity}, "
                             f"recompute says {recomputed_arity}")
            for chunk in ((0, 1) if arity == "CHUNKED" else (None,)):
                k = (model, target, chunk)
                if k in have:
                    continue
                sib = have.get((model, target, 1 - chunk)) if chunk is not None else None
                if sib is not None:
                    job, rank = sibling_rescue(sib[0], sib[1], chunk)
                    src = f"sibling {sib[0].name}"
                elif k in computed:
                    job, rank, src = computed[k], f"{trail:03d}", "recomputed"
                else:
                    raise SystemExit(f"cannot build {k}: plan of record says {arity} but the "
                                     "recomputation holds no such job to copy fields from")
                if covered(job):
                    continue
                if src == "recomputed":
                    trail += 1
                    if inferred:
                        src += ", arity INFERRED"
                tag = target + ("" if chunk is None else f"_c{chunk}")
                if a.first and a.first == f"{model}/{tag}":
                    rank, src = "000", src + ", pinned first"
                missing.append((f"job_{rank}_{model}_{tag}.json", job, src))

    for d in drift:
        print(f"DRIFT (plan of record used) {d}")
    if drift and any("INFERRED" in s for _, _, s in missing) and a.apply:
        raise SystemExit(
            "refusing to --apply: the recomputation disagrees with the plan of record on "
            f"{len(drift)} target(s), and this rescue also has to INFER the arity of a target "
            "with no local trace -- that inference is exactly what the disagreement discredits.")

    if not missing:
        print("nothing missing: every planned job is queued or already folded here")
        return
    for name, job, src in sorted(missing):
        print(f"{'WRITE' if a.apply else 'would write'} {name}  "
              f"n={job['n_samples']} seed={job['seed']} mps={job['mps']} "
              f"proj={job['proj_s']}s timeout={job['timeout_s']}s  [{src}]")
        if a.apply:
            p = qdir / name
            if p.exists():
                raise SystemExit(f"refusing to overwrite {p}")
            p.write_text(json.dumps(job, indent=1))
    print(f"{len(missing)} job(s), {sum(j['proj_s'] for _, j, _ in missing)/3600:.1f} "
          f"projected card-h" + ("" if a.apply else "  (dry run, use --apply)"))


if __name__ == "__main__":
    main()
