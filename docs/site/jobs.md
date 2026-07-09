# Jobs

Predictions and designs both create a **job**. This is how you track it, read
its output, and clean it up. All job endpoints are keyless; you can only reach
jobs you own (see [Authentication](authentication.md)).

## The Job object

```json
{
  "object": "job",
  "id": "6f1069cb2d665cb939f8baaa3cd261a6",
  "kind": "predict",              // "predict" | "design"
  "status": "running",            // queued | running | succeeded | failed | canceled
  "name": "myprotein",
  "model": "boltz2",              // null for designs
  "protocol": null,               // set for designs
  "progress": 0.5,                // 0..1, or null when indeterminate
  "stage": "msa",                 // human-readable current step
  "done": 0, "total": 1,          // sub-units completed / total
  "error": null,                  // set when status=failed
  "created_at": "2026-07-09T12:16:06Z",
  "started_at":  "...",
  "finished_at": null,
  "results_ready": false,
  "links": { "self": "...", "results": "...", "archive": "...", "logs": "..." }
}
```

## Poll one job

```bash
curl -s https://api.japanfold.com/v1/jobs/$JOB
```

Add `Prefer: wait=<seconds>` to block until the job finishes (or the timeout
elapses, capped at 60s; bare `wait` holds 25s) instead of returning immediately
— handy for short jobs:

```bash
curl -s -H 'Prefer: wait=60' https://api.japanfold.com/v1/jobs/$JOB
```

A simple poll loop:

```bash
until curl -s https://api.japanfold.com/v1/jobs/$JOB \
  | grep -qE '"status":"(succeeded|failed|canceled)"'; do sleep 5; done
```

## List your jobs

Paginated (cursor-based). `limit` defaults to 20 (max 100).

```bash
curl -s "https://api.japanfold.com/v1/jobs?limit=20"
```

```json
{ "object": "list", "data": [ { "object": "job", ... } ],
  "has_more": false, "next_cursor": null }
```

Pass `next_cursor` back as `?cursor=...` to fetch the next page.

## Cancel, delete

```bash
curl -s -X POST   https://api.japanfold.com/v1/jobs/$JOB/cancel   # stop a queued/running job
curl -s -X DELETE https://api.japanfold.com/v1/jobs/$JOB          # delete the job and its data
```

## Results

Once `results_ready` (or `status` is `succeeded`), read the scores and the list
of downloadable artifacts:

```bash
curl -s https://api.japanfold.com/v1/jobs/$JOB/results
```

A **prediction** result:

```json
{
  "object": "results",
  "job_id": "...",
  "kind": "predict",
  "ready": true,
  "rows": [
    { "id": "target_1", "status": "ok", "n_chains": 1, "n_residues": 33,
      "samples": 1, "msa": false, "plddt": 0.6675, "ptm": 0.3226,
      "runtime_s": 28.3 }
  ],
  "artifacts": [
    { "path": "target_1.cif", "target": "target_1", "type": "structure",
      "url": "/v1/jobs/.../artifacts/target_1.cif" }
  ],
  "archive_url": "/v1/jobs/.../archive"
}
```

- `rows` — one row per target, with confidence scores. Fields depend on the
  model and inputs: `plddt`/`complex_plddt`, `ptm`/`iptm`, and (Boltz-2 affinity
  runs) affinity fields. A rough read: interface `iptm` > 0.5, fold
  `complex_plddt` > 0.7.
- A **design** result carries the ranked `designs` instead of `rows`.
- `artifacts[].url` and `archive_url` are paths under the base URL — prefix them
  with `https://api.japanfold.com`.

## Download artifacts

One file:

```bash
curl -s https://api.japanfold.com/v1/jobs/$JOB/artifacts/target_1.cif -o target_1.cif
```

Everything as a zip:

```bash
curl -s https://api.japanfold.com/v1/jobs/$JOB/archive -o results.zip
unzip -oq results.zip -d results
```

## Logs

Plain-text run log — useful while a job runs or to debug a failure:

```bash
curl -s https://api.japanfold.com/v1/jobs/$JOB/logs
```
