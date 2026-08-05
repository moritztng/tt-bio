# JapanFold — Galaxy production deployment + production-readiness

Distilled from the deployment work on the prod Galaxy (~Jun–Jul 2026) plus the
on-box artifacts. The authoritative short version lives on the box at
`~/.japanfold-agent/RUNBOOK.md`.

JapanFold = the `aiand-bio` platform ("ai& Bio" / tt-bio serve) rebranded and put
into public production. It is presented as its own product ("Sovereign compute in
Japan"); ai& and Tenstorrent appear only as a "Powered by ai& on Tenstorrent AI
processors" credit.

Public hostnames, since the apex cutover on 2026-08-05:

| hostname | serves | origin |
|---|---|---|
| `japanfold.com`, `www.japanfold.com` | the marketing landing page | Cloudflare Pages (`japanfold-landing`) |
| `landing.japanfold.com` | 301 to the apex | Cloudflare Pages |
| `demo.japanfold.com` | the interactive SPA | this box, over the tunnel |
| `api.japanfold.com` | the published API base (`/v1/*`) | this box, over the tunnel |
| `docs.japanfold.com` | the docs site | GitHub Pages (`japanfold/japanfold`) |

Only `demo`, `api` and `ssh` still route through the tunnel. Landing assets are
never served off this box — that is deliberate, so a 15 MB hero video cannot
compete with predictions for the prediction host's bandwidth.

## The box
- Host **UF-EV-A13-GWH02** (a Tenstorrent Wormhole **Galaxy, 32 chips**,
  `/dev/tenstorrent/0..31`), 64-thread AMD EPYC 9354P, 566 GB RAM, 3.5 TB disk.
- Reachable from Moritz's pc as `ssh japanfold-ssh` (user **cust-team**), which
  goes **over the Cloudflare Tunnel** (`ssh.japanfold.com` → `ssh://localhost:22`,
  gated by a Cloudflare Access self-hosted app with a **service-token-only**
  policy — nothing reaches sshd without the token).
- This is the **prod JapanFold Galaxy** and is distinct from the customer-shared
  galaxy `bh-glx-exp-b03u14`. Passwordless `sudo` works here.
- **Port 8080 is taken** by a pre-existing service on this box (an RKE2 k8s
  cluster + nginx-ingress "Cluster Telemetry Dashboard"). JapanFold therefore
  runs the serve on **8090** and the fleet controller on **8770** (not the
  tt-bio defaults 8080/8765). Remember this on any fresh stand-up.

## Final architecture (three supervised layers + a maintenance agent)

1. **`japanfold.service`** (systemd, enabled) — the platform.
   `tt-bio serve --host 0.0.0.0 --port 8090 --controller-port 8770`, run as
   `User=cust-team`, `WorkingDirectory=/home/cust-team/mthuening`, venv
   `/home/cust-team/mthuening/tt-bio/env`. One `serve` master spawns the fleet
   **controller** (:8770) + **32 single-chip device workers** (one `spawn_main`
   per `/dev/tenstorrent/N`). Serves the built React SPA from `static/` on disk.
   - Env: `HF_HUB_CACHE=/home/cust-team/models`, `TT_METAL_LOGGER_LEVEL=FATAL`,
     `AIAND_BIO_SECURE_COOKIES=1`.
   - **WSGI = waitress** (32 threads, `ident="JapanFold"`), NOT the Flask dev
     server. `serve()` picks waitress in production and Flask's dev server only
     under `--debug`. Verify live with the `Server: JapanFold` response header
     (dev server would say `Werkzeug`).
   - **Graceful stop tuned to never wedge a chip**: `KillMode=mixed`,
     `KillSignal=SIGTERM`, `TimeoutStopSec=120`. SIGTERM hits only the main
     process; its `cluster.shutdown()` SIGINTs the workers for a clean device
     release; SIGKILL is a last resort far past the ~30s drain.
   - `Restart=on-failure`, `RestartSec=15`, crash-loop guard `StartLimitBurst=5 /
     StartLimitIntervalSec=600`.
   - `start-aiand.sh` now just delegates to `systemctl` (single source of truth,
     no accidental double-starts).

2. **`cloudflared-japanfold.service`** (systemd, enabled) — public ingress.
   Cloudflare Tunnel named `japanfold` → `localhost:8090`, config
   `~/.cloudflared/config.yml` (+ creds json). Ingress routes: `demo.japanfold.com`
   and `api.japanfold.com` → the server; `ssh.japanfold.com` → local sshd
   (Access-gated); catch-all → 404. The config also still lists `japanfold.com` and
   `www.japanfold.com`, left in place as an inert rollback target — no DNS points at
   them since the apex cutover. Auto-TLS at the Cloudflare edge
   (routed via Osaka/**kix** — on-brand for "sovereign compute in Japan"), no
   cert management on the box, no bandwidth caps. **Replaced ngrok entirely.**

3. **`japanfold-agent.service`** + **`japanfold-agent-sweep.timer/.service`** —
   the autonomous maintenance agent (see below).

Optional/manual: **`japanfold-db-warm.service`** (oneshot, `static`, NOT enabled
at boot by choice) warms the UniRef30 MSA index into page cache.

## The maintenance agent
- A **resumed Claude Code session** (`claude --resume eaab7bb1 --model
  claude-sonnet-5 --dangerously-skip-permissions`) running inside a **tmux**
  session `japanfold-agent`, supervised by `japanfold-agent.service`
  (`launch.sh` keeps the tmux alive; `run-agent.sh` respawns claude if it exits).
  Resuming *this* session was deliberate: it carries the entire deployment
  history/context, strictly better than seeding a fresh agent with a runbook.
- **Chat with it**: `tmux attach -t japanfold-agent` (detach `Ctrl-b d`).
- **Sweeps are pushed in, not self-scheduled**: `japanfold-agent-sweep.timer`
  (`OnBootSec=10min`, `OnUnitActiveSec=6h`, `Persistent=true`) runs `sweep.sh`,
  which waits until the agent's tmux pane is byte-stable (idle) before
  `tmux send-keys` of `sweep-prompt.txt` + Enter. **Note the drift**: the timer
  fires **every 6h**, but `RUNBOOK.md`'s prose still says "every 30 minutes"
  (an earlier value). The timer is ground truth.
- Autonomy boundary (**conservative, decided**): auto-fix a dead service (drain
  first), free disk, soft-reset a wedged chip via the ladder, re-fetch a corrupt
  checkpoint, let the supervisor respawn dropped workers. **Never autonomously**:
  full host reboot, anything that drops the box, or `kill -9` a worker mid-op.
- It logs every sweep/action to `~/.japanfold-agent/maintenance.jsonl` and calls
  `~/.japanfold-agent/notify.sh` on ANY anomaly (even self-fixed). A fully clean
  sweep logs only, no notify (keeps alerts signal). `notify.conf` selects the
  channel (slack/discord/telegram/webhook); it also always records locally.
  `recover.sh` reclaims a stray manual resume back into the systemd session.

## Observability
- `~/.aiand-bio/events.jsonl` — append-only JSONL of every platform event
  (`server_started`, `job_submitted/started/done/killed/canceled`,
  `job_rejected` with the limit that fired). Session ids and client IPs are
  short-hashed so it's safe to paste into an LLM ("what happened in the last
  hour / who's hitting limits"). Size-rotated (`.jsonl` + `.jsonl.1`).
- Fleet logs: `~/.aiand-bio/jobs/_cluster/workers.log` + `controller.log`.
- `curl -s localhost:8090/api/cluster` → `online_workers` (expect 32),
  `controller_alive`. `curl -fsS https://api.japanfold.com/api/health` → `status ok`.
  Do not health-check the apex: it is a static Pages site and stays up when this box
  is dead.

## Stand it up again from scratch (order matters)
1. **Serve first, on 8090/8770** (8080 is taken by k8s). Confirm 32 workers:
   `curl localhost:8090/api/cluster` → `online_workers: 32`.
2. **Install cloudflared** (official binary to `/usr/local/bin`). One interactive
   step is the human's: `cloudflared tunnel login` → authorize the `japanfold.com`
   zone in a browser → writes `~/.cloudflared/cert.pem`.
3. `cloudflared tunnel create japanfold` (writes a creds json — keep secret),
   `cloudflared tunnel route dns japanfold demo.japanfold.com` (auto-creates the CF
   DNS record; repeat for `api`), write `~/.cloudflared/config.yml` mapping the
   hostnames → `http://localhost:8090`, `cloudflared tunnel ingress validate`.
   Leave the apex and `www` alone — Cloudflare Pages owns them.
4. Wrap the tunnel in `cloudflared-japanfold.service`, `enable --now`.
5. Wrap the serve in `japanfold.service` (waitress + the graceful-stop settings
   above), `enable --now`. Cut over from any hand-started serve **only while
   idle** (`runs.running == 0`) and **gracefully** (SIGTERM/SIGINT, never
   `kill -9`) — see the wedge lesson below.
6. (Optional) the maintenance-agent + sweep timer units, and the db-warm unit.
7. Provision MSA: `tt-bio msa --db uniref30 --path /data/colabfold_db`
   (installs pixi/localcolabfold + downloads/indexes UniRef30, ~500 GB, multi-hour;
   the `UNIREF30_READY` sentinel marks it done → MSA runs offline, sequences stay
   in-cluster).

## Hard-won lessons / gotchas ("that's exactly what happened")
- **The cardinal sin: `kill -9` a worker mid-op wedges its chip**, and only a
  host reboot recovers it. Always **drain** (`runs.running == 0`) then stop
  **gracefully** (SIGTERM → `cluster.shutdown()` SIGINTs workers → clean device
  release). An idle SIGKILL is survivable (verified: chips stayed healthy); a
  **mid-op** SIGKILL is the hazard.
- **systemd restart still mass-SIGKILLs on shutdown** (observed ~1677 SIGKILLs
  in one stop). `KillMode=mixed`+SIGTERM sends TERM only to the main; if workers
  don't all exit within `TimeoutStopSec`, systemd SIGKILLs the rest. This is
  tolerable **only because restarts happen at idle** — a residual rough edge, not
  a fully clean graceful drain.
- **The device-init race (root cause, FIXED 2026-06-29).** Some Protenix jobs
  failed instantly with `mesh_device.cpp:993 SubDeviceManagerTracker is not
  initialized … contains only remote devices (no local device)`. Root cause: the
  host-wide `_device_init_lock` in `tenstorrent.py` had become **dead code** (its
  only opt-in env gate vanished when design moved in-process), so all 32 worker
  startup opens raced the non-concurrency-safe UMD init and some chips came up
  "remote-only" (no local dispatch) → first program dispatch of ANY model throws;
  Protenix just exposed it first. Fix (engine, both repos): (1) **always**
  serialize device open/close host-wide (removed the vestigial gate) — the 32
  startup opens happen one-at-a-time; verified this did NOT slow boot (32/32 in
  ~25s); (2) `_assert_local_dispatch(dev)` runs one trivial op right after each
  open to catch a remote-only bring-up; (3) a failed check closes+raises → the
  worker exits → the pool supervisor respawns it → the respawn reopens serialized
  = clean. A `state.reset()+reopen` reactive **retry** was the earlier trigger and
  was **removed** (reopen under concurrency landed on a remote-only slot).
- **Design-shard deadlock** was fixed by running BoltzGen design **in-process**
  in the worker (reusing its persistent device) instead of cold-opening a chip
  per shard subprocess. Concurrent cold-opens deadlocked on the UMD robust mutex
  during `LocalChip::start_device` — the contention is open-vs-runtime, not just
  open-vs-open.
- **A reboot is the real fix for a wedged chip / ARC-stuck chip**, but it drops
  the whole box and the agent — it's a **human decision**, never autonomous. In
  practice a reboot cleanly recovered a chip (tt27) that had been wedged for days
  and soft resets couldn't clear.
- **Reboot recovery works end-to-end**: after a full host reboot all four
  services came back on their own, hardware re-enumerated 32 chips, 32/32 workers,
  site 200, HTTP→HTTPS still enforced, deployed code confirmed live. The one
  expected non-issue: **MSA page cache is cold** (~5 GB vs ~430 GB) — first few
  MSA predictions are slow (100–160s vs ~8s warm) until it rewarms organically or
  you run the db-warm script. NFS ceiling ~300 MB/s; full warm ~20–24 min; more
  than ~48-way parallelism doesn't help.
- **Disk: judge by absolute free space, not %.** 3.5 TB box normally ~240 GB
  free = "93% used" is normal, do nothing; only act below ~20 GB free.
- **cloudflared setup**: `tunnel login` is the only interactive step (needs a
  browser). If bringing a custom domain on **ngrok** instead, the CF DNS record
  must be a **CNAME set to "DNS only" (grey cloud)** or CF's proxy collides with
  ngrok's TLS — and a custom ngrok domain may need a paid plan. Cloudflare Tunnel
  has neither problem, which is why it won.
- **ngrok is dead** but its old authtoken is leaked in `~/.bash_history` — should
  be rotated. The reboot wiped the in-shell `NGROK_AUTHTOKEN`, which is what broke
  the old tunnel and forced the (better) Cloudflare Tunnel migration.
- **Two-repo model**: engine code (`tt_bio/*.py`) → both `tt-bio` (branch `main`)
  and `aiand-bio` (branch `main`); platform code
  (`tt_bio/platform/*`) → `aiand-bio` only. `tt_bio` is pip-installed **editable**
  pointing at the aiand-bio checkout, so the running service imports aiand-bio
  sources. **Frontend** changes go live after `npm run build` (no restart);
  **Python** changes need a serve restart at an idle window.
- **Deploy = pull + restart.** Pushing a fix does NOT deploy it; the running
  service keeps old code until the checkout is pulled and the service restarted
  (at idle). Verify the running process actually reloaded by grepping the fixed
  code on disk AND confirming a restart post-dates the commit.
- **The sweep prompt can be lost if injected while the agent is still loading**
  the ~27 MB resumed session — hence `sweep.sh`'s idle-readiness wait. That's
  "exactly what happened" to the first post-boot sweep.

## PRODUCTION-READINESS — honest verdict (as of 2026-07-12)

**Verdict: production-ready for a public research demo, running live and healthy
(32/32 workers, site 200, 0 failed jobs across days of 6h sweeps). Not
hardened to a paid-SLA / high-abuse standard — a few known-fragile edges remain,
and recovery from a hard fault is partly manual.**

What IS production-ready (done + verified live):
- **Supervised, boot-surviving stack**: serve, tunnel, and agent are all systemd
  units, enabled, auto-restart on crash, and verified to come back after a full
  reboot.
- **Real WSGI** (waitress) — not the Flask dev server. Handles the polling load.
- **Stable HTTPS ingress** on a real domain via Cloudflare Tunnel; HTTP→HTTPS
  enforced; secure cookies on.
- **Abuse backstops**: per-session (3 concurrent / 12 submits-min), per-IP, and
  global (64 active) caps → friendly 429s; 8 MB body cap; per-job ceilings
  (≤1024 residues, ≤10 structures, ≤50 designs, ≤500 steps) + 25/45-min runtime
  watchdogs. Input validation + per-session ownership + error sanitization.
- **Privacy/sovereignty holds**: MSA runs against the offline `/data` UniRef30 DB
  (sequences stay in-cluster; no external MSA dependency under load).
- **Observability**: structured `events.jsonl` + fleet logs + `/api/cluster`.
- **Device-init race fixed at root**; concurrent-open deadlock fixed; graceful
  shutdown that (at idle) doesn't wedge chips.
- **A self-healing worker pool** (respawns dropped workers) and an autonomous
  maintenance agent doing conservative auto-fixes + alerting.

What is fragile / known-risk:
- **Chip/ARC wedge → needs a human reboot.** A mid-op kill (or an ARC-stuck
  chip) can only be cleared by rebooting the whole box, which is deliberately
  NOT autonomous. So the worst-case recovery is manual and drops the site.
- **Restart shutdown still mass-SIGKILLs workers** (safe only because it's done
  at idle) — not a clean graceful drain yet.
- **Single host, no redundancy**: one Galaxy, no failover; any reboot/outage =
  full downtime until it comes back (it does come back cleanly, but there's a gap
  and a cold MSA cache after).
- **Notify channel** depends on `notify.conf` being configured (else alerts are
  local-only).
- **RUNBOOK/timer drift**: RUNBOOK says 30-min sweeps, timer is 6h — so anomaly
  detection latency is up to ~6h, not 30 min.

What is still manual / deferred:
- **MSA cache warm after reboot** is manual (db-warm is `static`, not enabled at
  boot) — intentional, to protect NFS bandwidth, but means slow first predictions
  post-reboot unless someone runs it.
- **Per-IP limiting + logging should switch to `CF-Connecting-IP`** (currently
  rightmost `X-Forwarded-For`, tuned for the old ngrok path).
- **Not yet done** (flagged, lower priority): tighten CORS (currently
  allow-any-origin from the Vite dev days), rate-limit GET/read endpoints, a
  sustained mixed-load soak test, a friendlier "at capacity" UI state, a
  data-handling notice in the banner, and real uptime monitoring/alerting on
  `/api/health` (the event log is post-hoc only).

## Files (on the prod box)
- `~/.japanfold-agent/` — RUNBOOK.md, launch.sh, run-agent.sh, sweep.sh,
  sweep-prompt.txt, recover.sh, db-warm.sh, notify.sh, notify.conf + .template,
  maintenance.jsonl.
- `/etc/systemd/system/` — japanfold.service, cloudflared-japanfold.service,
  japanfold-agent.service, japanfold-agent-sweep.{service,timer},
  japanfold-db-warm.service.
- `~/.cloudflared/config.yml` (+ tunnel credentials json — secret).
- Serve checkout at `/home/cust-team/mthuening/aiand-bio` (venv `../tt-bio/env`).
