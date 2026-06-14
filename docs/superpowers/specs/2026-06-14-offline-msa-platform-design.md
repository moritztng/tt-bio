# Offline MSA for the ai& Bio platform

**Date:** 2026-06-14
**Status:** Approved

## Problem

The platform generates MSAs via the public ColabFold API
(`--use_msa_server`), which sends every input sequence to
`api.colabfold.com`. On this deployment, **sequences must not leave the
cluster**, so the platform must generate MSAs offline against a local
database, while still scaling MSA generation across all CPUs of every
galaxy in the fleet.

## What already exists

- **Engine (`tt-bio` main):** offline MSA is fully implemented.
  `compute_msa_offline()` runs `colabfold_search <query> <db> <a3m>` and is
  invoked automatically by every worker when `--msa_db_path` is set (Boltz-2,
  ESMFold2, and protenix-v2 MSA stages all route through it). `tt-bio msa`
  downloads/indexes the ColabFold UniRef30 DB.
- **Platform (`aiand-bio`):** `jobs.py:_build_cmd` unconditionally appends
  `--use_msa_server` when the "Generate MSA" toggle is on — the only thing
  forcing the external service.
- **Infra:** `/data` is a shared, writable, multi-host NFS mount
  (`172.16.102.239:/srv/nfs/multihost`, 14 TB, ~1.2 TB free) present on
  every galaxy. It is the natural home for one shared MSA DB.

## Design

### 1. Engine — cap `colabfold_search` threads (commit to `tt-bio` main)

`compute_msa_offline` currently passes `--threads <os.cpu_count()>`. With
many per-worker MSAs running on one host that oversubscribes the CPU (the
same thrash already fixed for folding). Change `--threads` to honor the
existing per-worker cap:

```python
threads = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 1)
```

`_spawn_worker_processes` already sets `OMP_NUM_THREADS = cores // workers`,
so concurrent MSAs on a host share cores cleanly. Single-run CLI is
unchanged (env unset → all cpus). This is the only engine change and is
what makes "leverage all CPUs across ≥2 galaxies" correct rather than
thrashing. Merged into `aiand-bio` afterward.

### 2. Platform — offline routing (commit to `aiand-bio`)

`tt-bio serve` gains two options:

- `--msa-db-path` (default `/data/colabfold_db`, env `AIAND_BIO_MSA_DB`):
  the shared ColabFold DB.
- `--msa-mode {auto|offline|server}` (default `auto`):
  - `auto` — offline when `<db>/UNIREF30_READY` exists, else the public
    server. **This is the "server stays as a fallback" behaviour**: once
    the DB is staged, every run is offline; until then (or on a galaxy
    without the DB) it falls back to the server.
  - `offline` — always offline (hard sovereignty; predict errors if the DB
    is missing).
  - `server` — always the public API (break-glass / legacy).

Threaded `serve() → create_app() → JobManager`. `_build_cmd` resolves the
mode **per job** (a cheap `UNIREF30_READY` stat), so the moment the
download finishes, new jobs go offline with no server restart:

```python
if p.get("use_msa_server"):           # the "Generate MSA" toggle
    db = self._msa_db()               # path if offline applies, else None
    cmd += (["--msa_db_path", db] if db else ["--use_msa_server"])
```

`_msa_db()`:

```python
def _msa_db(self):
    if self.msa_mode == "server" or not self.msa_db_path:
        return None
    if self.msa_mode == "offline":
        return self.msa_db_path
    return self.msa_db_path if (Path(self.msa_db_path) / "UNIREF30_READY").exists() else None
```

### 3. Multi-galaxy — no new code

`/data` is mounted on every galaxy, so all share one DB. The controller
already fans a batch's structures across all connected galaxies; each runs
offline MSA locally against the shared DB with its capped CPU threads → all
galaxies' CPUs are used across the batch. The master's
`_validate_offline_msa_db` check works because `/data` is shared, so it sees
the same DB the workers do.

### Sovereignty guarantee

In `auto`/`offline` mode, when the DB is present a per-request search
failure does **not** silently fall back to the public server (which would
leak the sequence) — the job errors instead. Server fallback only applies
when no DB is resolved (e.g. before the download finishes, or a galaxy
without the mount).

## Provisioning

One-time: `tt-bio msa --db uniref30 --path /data/colabfold_db` (downloads +
indexes UniRef30, ~500 GB, fits in the 1.2 TB free on `/data`). The DB is
public reference data, so the download itself over the internet is fine;
only user sequences are sovereign, and offline MSA keeps those in-cluster.

## Commit split

| Change | Repo |
|---|---|
| `compute_msa_offline` thread cap | `tt-bio` main → merge into `aiand-bio` |
| `serve` flags, `create_app`/`JobManager` threading, `_build_cmd` routing | `aiand-bio` (platform-only module) |

## Out of scope (YAGNI)

- A dedicated MSA pre-pass/pool (batch all sequences into one `mmseqs` per
  galaxy). The per-worker path already saturates CPUs via fan-out.
- EnvDB (UniRef30 only, per decision).
