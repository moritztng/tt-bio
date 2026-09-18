# s4 was killed by the orchestrator, not reaped by a systemd scope

`c12-compose-fold` pass 36 concluded a durable infra lesson from s4's death:

> **`setsid nohup` does not survive its launching ssh session's systemd scope reap on qb2** — long
> unattended jobs need `systemd-run --user --unit=`, and `KillUserProcesses`/`Linger` being correct
> is not sufficient.

**That is refuted by the journal it cites.** `session822.journal.txt`, read from qb2:

    21:55:44  New session 822 of user ttuser
    21:55:44  Started session-822.scope
    21:55:45  Session 822 logged out. Waiting for processes to exit.     <- did NOT kill them
    22:03:58  session-822.scope: Deactivated successfully.
    22:03:58  Consumed 8min 15.422s CPU time, 1.9G memory peak

Two things in that trace contradict the conclusion:

1. **`systemd-logind` logged "Waiting for processes to exit" one second after login** and then did
   nothing for **8 minutes 13 seconds**. That is `KillUserProcesses=no` working exactly as
   configured. Had the scope reaped s4, the kill would be at 21:55:45, not at 22:03:58.
2. **"Deactivated successfully" is the message for a cgroup whose last process has exited.** It is
   the *effect* of the process dying, not the cause. The `Consumed 8min 15.422s` line is s4's own
   lifetime accounting being flushed at teardown.

**What actually killed s4: the orchestrator's SIGTERM at ~22:03:58.** The orchestrator cleared it
(pids 162379/162393/162485/162504) while diagnosing dev2, and the scope deactivated in the same
second because that was its last process. The 8 min 13 s the scope stayed up *after logout* is
positive evidence that `setsid nohup` DID survive the ssh logout on this host.

**Why s4 produced zero folds: dev2's device-open was wedged.** Independently proven, and not by s4:

    bare get_device() on dev2, BEFORE   exit 124, timed out at 420 s, twice, nothing past ttnn.CONFIG
    tt-smi -r 2 (as the owning user)     41.07 s, per-chip
    bare get_device() on dev2, AFTER    device opened in 3.7 s

s4 was stalled at device open on an unopenable card, which is why it was "still loading" at 8
minutes. 114x is not a slow load.

**No fleet-wide switch to `systemd-run --user --unit=` is warranted on this evidence.** It may still
be better practice for other reasons; it was not the failure here, and the 8 min 13 s of post-logout
survival is evidence against the stated mechanism.

## The process failure that minted the wrong lesson, which is the transferable part

`OWNER-qb2-card2.txt` is the file `c12-compose-fold` wrote to guard its session. It lives in
**qb2's** `state/`, names the exact pid the orchestrator killed, and sets its own removal condition
("Remove this file after the session ends or if pid 24290 is gone"). The orchestrator checked pc's
`state/leases/`, `fuser` and `ps` — and never read it. The kill was still inside the file's own terms
(24290 was a corpse with no forward progress, and the file's stated end was ~21:42Z against an action
at ~21:47Z), but it was taken without consulting the fleet's designated coordination artifact.

**The general failure: when two agents share a host, the victim of a cross-agent kill diagnoses its
own corpse without knowing it was killed.** `c12-compose-fold` did the forensics carefully — it even
declined to attribute s3's earlier death to the same mechanism, which was the right call — and still
landed on a wrong mechanism, because the one fact that explained everything (another agent SIGTERMed
it) was not visible in any artifact it owned. A kill by one agent must be *written where the victim
will look*, not just executed.

Two concrete consequences:
- Check `state/OWNER-*` **on the host that owns the card**, not only the orchestrator's lease dir,
  before signalling anything.
- After killing another row's process, append the kill to that row's state doc or brief. The cost of
  not doing it here was one wrong durable lesson and a wasted session.
