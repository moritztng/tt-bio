# Running tt-bio on many machines

tt-bio runs one machine at a time. On each machine a controller listens on 127.0.0.1
and a worker per chip takes jobs from it. Whatever spreads work across machines sits on
top and talks to each machine's controller. That can be JapanFold, Slurm, Ray or
[a fifty-line script](../examples/many_hosts.py). This page is the contract that layer
builds against.

```
  your scheduler: queue, accounts, fairness between users, which machine gets what
        |                 |                 |
   host A             host B             host C          (ssh, or an agent on each host)
   controller         controller         controller      127.0.0.1 only
   worker per chip    worker per chip    worker per chip
```

No tensor crosses machines. Every job (a target to fold, a design shard, a batch to
embed) runs whole on one chip, so the layer on top only has to decide where each job
goes.

## Setting up one host

```bash
tt-bio controller --port 8765                        # starts a worker on every chip
```

Or start the workers yourself, which is what a platform does when it wants to own each
chip's lifecycle (reset a chip, keep it out of service):

```bash
tt-bio controller --port 8765 --no-local-workers
tt-bio worker --connect http://127.0.0.1:8765 --device_ids 0     # one per chip, or omit for all
```

The controller binds 127.0.0.1 and nothing else. It has no authentication, so reach it
from a process on the same host or through ssh.

## Submitting work

Submit with the same commands you run locally, pointed at the controller:

```bash
tt-bio predict ./targets --model boltz2 --out_dir ./out \
    --controller http://127.0.0.1:8765 --run-id job-42 --owner 3f9a...
```

`predict`, `embed`, `saprot` and `design` all take `--controller`. The command checks
the inputs, builds the run, blocks until every job has ended and writes results to
`--out_dir` exactly as a local run would. It exits 0 when every job succeeded, 2 when
some failed and 1 when all did.

- `--run-id` (on `predict` and `design`) names the run, so you can watch or cancel it by
  that id.
- `--owner` is an opaque fairness key. The controller shares chips so that the owner
  holding the fewest gets the next free one, and a lone owner gets them all. Pass a hash
  of a session or account id, never the id itself and never a secret. tt-bio has no
  notion of who pays; that stays in your layer.

While the command runs, these reads are free to poll:

| Request | Answer |
|---|---|
| `GET /cluster` | what the host advertises (below) |
| `GET /runs/<run>/jobs` | `{"jobs": [{"id", "status", "stage"}]}`, one per input; `stage` is `prepare`, `msa`, `fold`, `score` or `save` |
| `GET /runs/<run>/status` | `{"status": "running" \| "ok" \| "failed" \| "canceled" \| "missing"}` |
| `GET /runs/<run>/events?after=<seq>` | up to 500 progress events after `seq`, plus the run's totals |
| `GET /runs/<run>/results` | the result row of every finished job |
| `POST /runs/<run>/cancel` | stops new leases and marks unfinished jobs canceled; a job already on a chip runs to its end and its result is dropped |
| `GET /healthz` | `{"ok": true}` |

A run's status is `ok` only if every job was `ok`, and `failed` if any was not.

## What a host advertises

`GET /cluster` is the host's answer to "what can you run right now":

```json
{"online_workers": 31, "total_workers": 32,
 "workers": [{"worker_id": "gwh01:tenstorrent:7", "host": "gwh01", "accelerator": "tenstorrent",
              "device_id": "7", "label": "gwh01:tt7", "model": "esmfold2",
              "online": true, "idle_s": 0.4, "running": [{"run_id": "job-42", "job_id": "t3"}]}],
 "hosts": [{"host": "gwh01", "devices": 31, "accelerators": ["tenstorrent"]}],
 "runs": {"running": 1, "ok": 17}, "jobs": {"running": 3, "ok": 20}}
```

- **`online_workers` is the number of usable chips**, not the number installed. A worker
  registers only after its chip opens, so a chip that fails to come up never appears. On
  the four Galaxies this page was tested on, one reported 31 of 32.
- **`online`** means heard from in the last 20 s. An idle worker asks for work every
  second and a busy one heartbeats from its own thread, so a worker that goes quiet has
  died or its host has.
- **`model`** is the model the worker has loaded. The controller prefers to give a worker
  jobs for the model it already has, so it does not reload.
- **`running`** is the jobs it holds right now. An empty list on an online worker is a
  free chip.

To weigh machines against each other, divide their work by `online_workers`.

## The worker protocol

Read this part if you want to replace tt-bio's controller with a scheduler of your own on
the host and keep tt-bio's workers. A worker makes four kinds of POST. A submitting
command makes one more, `POST /runs`, and polls the reads above, so a replacement answers
those too.

**`POST /runs`** `{"run_id"?, "data", "out_dir", "result_dir", "owner", "config", "jobs": [{"id", "name", "input_b64"}]}`
creates a run and answers `{"run_id", "total"}`. Treat `config` as opaque and hand it to
workers unchanged; it is the model configuration the submitting command built.

**`POST /lease`** `{"worker": {...}, "batch_size": 1}` answers
`{"run_id", "config", "jobs": [{"id", "name", "input_b64"}], "lease_s"}`, or
`{"jobs": [], "lease_s"}` when there is nothing to do. Every job in one answer belongs to
one run. `worker` is the advertisement above: `worker_id`, `host`, `accelerator`,
`device_id`, `label` and `model`. A worker asks only when it holds nothing, so a job
still leased to the asking worker was abandoned (its process restarted, or its
completion never arrived) and goes back in the queue.

**`POST /heartbeat`** `{"worker": {...}}` answers `{"ok": true, "lease_s"}`. It marks the
worker alive and renews the lease on every job it holds.

**`POST /events`** `{"run_id", "worker_id", "event": {...}}` is progress. It updates the
job's `stage` and the worker's liveness, but renews no lease. Workers send it and ignore
failures.

**`POST /complete`** `{"run_id", "worker_id", "result": {"id", "status", "error"?, ...}, "event", "outputs"?}`
settles one job. `status` is `ok` or `failed`. `outputs` maps each file's path relative to
the output directory to its base64 bytes. A worker that proved it shares the submitter's
filesystem writes the file in place and sends `"tt-bio-shared-path:<path>"` instead.

## Leases

A lease is a promise with an expiry. It lasts `lease_s`, 120 s unless you start the
controller with `--lease-s`, and a worker renews it every `lease_s / 12`, which is every
10 s. The worker takes the interval from the `lease_s` in each answer, so the controller's
flag is the only setting.

If a worker dies, its heartbeats stop and its jobs go back in the queue `lease_s` later,
keeping their place in line. The next worker to ask gets them. The heartbeat runs on its
own thread, so a job may run far longer than its lease without being handed out twice.
On four Galaxies folding eight inputs, one of them held for 20 minutes, the controllers
recorded 540 renewals: 8.1 s apart at the median, and never more than 14.2 s, with
heartbeats then 8 s apart. The same stall at 10 s apart reads about 16 s, seven times
inside the lease. Every job finished on its first attempt
([measurement](../perf/dst_boundary/)).

The controller has no attempt limit: a job goes back as often as the worker holding it
dies. Put a limit in your layer if you want one. JapanFold fails a job after two tries.

JapanFold's own API uses the same numbers one level up: a host's agent leases a job from
the API for 120 s and reports on it every 10 s. A lost worker or a lost host is noticed
within two minutes either way.

## Settling exactly once

A result counts only if it comes from the worker holding the job's lease when it
arrives. The controller settles with a single conditional update, "this job, held by
this worker, still running", and ignores a completion that matches nothing. Four things
follow:

- A worker whose lease lapsed and whose job went to another worker cannot overwrite the
  second worker's result when it finishes late.
- Sending the same completion twice changes nothing, so a worker can retry `/complete`
  after a network error, and tt-bio's worker does.
- A canceled run stays canceled. A completion cannot reopen it.
- A run ends once. The `run_done` event is written once, by the completion that ended
  the last job.

The mistake to avoid is settling by job id alone. Then a slow worker's late result
replaces the one that counted, or a job that ran twice is counted twice. If you charge for
work, charge once per run when it reaches a final status, keyed by the run id, not once per
completion you receive.

## Across machines with no platform

[`examples/many_hosts.py`](../examples/many_hosts.py) is the whole layer for a small
lab. It reads `online_workers` from each host over ssh, gives each host a share of the
inputs in proportion, runs `tt-bio predict --controller` there, and copies the results
back. A host that fails hands its share to the others.

```bash
python examples/many_hosts.py ./targets ./out --model esmfold2 \
    --host ubuntu@galaxy1:8765 --host ubuntu@galaxy2:8765 \
    --host ubuntu@galaxy3:8765 --host ubuntu@galaxy4:8765
```

On four Wormhole Galaxies with 127 usable chips between them it folded twelve inputs
of 120 to 1300 residues, three per host, all twelve `ok`, and in a second run eight more
in 21 minutes, two per host, all `ok`.

A service with many users wants the other direction: an agent on each host pulls jobs
from a central queue, submits each to its own controller with `--run-id` and `--owner`,
and reports progress from `GET /runs/<run>/jobs`. That is how JapanFold runs.
