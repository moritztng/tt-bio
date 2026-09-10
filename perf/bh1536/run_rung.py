#!/usr/bin/env python3
"""One (model, token-count) rung of the Blackhole 1536 ladder.

Runs the fold as a subprocess, then judges it on the ARTIFACT: a CIF with the
expected number of residues in it, and a results.json whose token count matches
the fixture. An exit status of 0 is not accepted as evidence, and neither is a
"status": "ok" field -- both have shipped a green run over an empty output
folder before. Peak host RSS and MemAvailable floor are sampled while it runs,
because a host OOM has taken one of these boxes down.

Verdicts: PASS (artifact checked and scored), OOM (an allocator refusal the run did not
survive), STALLED (stopped writing to its log for --stall seconds; the last line it wrote is
kept, because where it stopped is the diagnosis), TIMEOUT (still writing, but killed at
--budget), FAIL (ran and produced no usable artifact), and CONTENDED, which is NOT a result --
a co-tenant held card 0, so this rung measured nothing and has to be walked again.

Appends one JSON object per rung to results.jsonl and one line to sweep.log.
"""
import argparse, fcntl, json, os, re, resource, shutil, signal, subprocess, sys, time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from tt_bio.device_lease import CONTENDED_EXIT_CODE  # noqa: E402  (75, not re-typed here)
from tt_bio.main import AFFINITY_MODELS  # noqa: E402  (the registry, not a literal list here)
OUTROOT = WT / "perf" / "bh1536"
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"

# The refusals we classify. DRAM and L1 are different walls with different fixes.
# The parenthetical is the whole classification and is captured on purpose. `free` vs
# `largest free block` is what separates the two OOM classes the campaign asks to be told
# apart: request > bank size is one oversized tensor, request < free but > largest free block
# is fragmentation, and request > free is plain exhaustion. Dropping it (as a regex that stops
# at "bank size is N B" does) makes all three look identical.
_ALLOC = (r"Not enough space to allocate (?P<req>\d+) B {kind} buffer across (?P<banks>\d+) "
          r"banks, where each bank needs to store (?P<per_bank>\d+) B, but bank size is "
          r"(?P<bank_size>\d+) B\s*\(allocated: (?P<allocated>\d+) B, free: (?P<free>\d+) B, "
          r"largest free block: (?P<largest>\d+) B\)")
OOM_PATTERNS = [
    (re.compile(_ALLOC.format(kind="DRAM")), "dram"),
    (re.compile(_ALLOC.format(kind="L1")), "l1"),
    (re.compile(r"grow to (?P<req>\d+) B .*?beyond max L1 size of (?P<bank_size>\d+) B", re.S),
     "l1_cb"),
]


def classify(g: dict) -> str:
    """Which of the two OOM classes this refusal is, from the allocator's own numbers."""
    per_bank, bank = g.get("per_bank"), g.get("bank_size")
    if per_bank is None:                       # a circular-buffer throw carries no bank figures
        return "l1_static_cb"
    if per_bank > bank:
        return "oversized_tensor"              # no chip state would have served this request
    free, largest = g.get("free"), g.get("largest")
    if free is None or largest is None:
        return "residency_unclassified"
    if per_bank > free:
        return "residency_exhausted"           # the chip is simply full
    if per_bank > largest:
        return "fragmentation"                 # room exists, no single block holds it
    return "unclassified"


#: A rung that never opened the card measured nothing, and recording it as FAIL publishes a
#: Blackhole ceiling that no Blackhole allocator set. Five tasks share these boxes tonight and
#: one of them runs an UNPINNED pytest, which brings up every visible chip and holds card 0's
#: lease: protenix-v1 at 1536 was scored FAIL at 23:45Z purely because
#: tt-boltz-kisoji/env/bin/python -m pytest (pid 789447) had the lease for its whole 120 s wait.
#: Same class as capacity_gate's _host_killed and _input_rejected — the thing under test has to
#: have run before its verdict means anything.
_CONTENDED = re.compile(r"is in use by pid|leased by another process|Refusing to open it "
                        r"concurrently|DeviceInUseError")


def contended(returncode: int, text: str) -> bool:
    return returncode == CONTENDED_EXIT_CODE or bool(_CONTENDED.search(text or ""))


#: The card would not come up at all. Different cause from contention, same consequence: the fold
#: never reached the model, so the rung measured nothing about capacity. opendde:1536 stalled and
#: left the chip in a state where `risc_firmware_initializer.cpp:1115` threw on the next open, and
#: opendde-abag AND protenix-v1 were both then recorded FAIL at 1536 -- two published Blackhole
#: capacity results from a card that never executed an instruction. `tt-smi -r 0` cleared it.
_WEDGED = re.compile(r"device open failed"
                     r"|risc_firmware_initializer"
                     r"|Timed out while waiting for active ethernet core"
                     r"|contains only remote devices")


#: SIGBUS in a worker means the memory-mapped card went out from under it, which is a card fault
#: and not a capacity result. openfold3:512 was recorded FAIL at 14.2 s on `SpawnProcess-1 exit -7`
#: because `tt-smi -r 0` cleared the wedge left by opendde while that rung had the chip mapped.
#: SIGSEGV is deliberately NOT here: that one can be a real software bug and must stay visible.
_WEDGED_SIGNALS = (-7, 135)


def device_wedged(text: str, returncode: int = 0) -> bool:
    return returncode in _WEDGED_SIGNALS or bool(_WEDGED.search(text or ""))


def not_a_measurement(returncode: int, text: str) -> str | None:
    """"CONTENDED" / "WEDGED" if this rung never reached the model, else None.

    One predicate for the whole class, because it has now bitten twice with two different
    causes. The question a rung has to answer is "what does this silicon do at this size", and
    a run that never got a working card has not answered it either way -- recording FAIL there
    publishes a ceiling nothing measured.
    """
    if contended(returncode, text):
        return "CONTENDED"
    # The dispatcher reports a worker's exit code in its own message, so a worker killed by a
    # card fault is visible even though run_rung only ever sees the PARENT's return code.
    if re.search(r"SpawnProcess-\d+ exit (?:-7|135)\b", text or ""):
        return "WEDGED"
    if device_wedged(text, returncode):
        return "WEDGED"
    return None


def sample_host():
    rss = 0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            st = (p / "statm").read_text().split()
            cmd = (p / "cmdline").read_bytes()
        except OSError:
            continue
        if b"tt_bio" in cmd or b"python" in cmd:
            rss += int(st[1]) * resource.getpagesize()
    avail = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
    return rss, avail


def cif_residues(path: Path) -> int:
    """Distinct (chain, seq id) in the atom records. Reads the structure, not a header."""
    seen = set()
    text = path.read_text(errors="replace")
    if "_atom_site." in text:                      # mmCIF loop
        cols, rows = [], []
        in_loop = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1]); in_loop = True; continue
            if in_loop:
                if s.startswith("#") or s.startswith("_") or s.startswith("loop_"):
                    if rows: break
                    in_loop = False; cols = []; continue
                if s: rows.append(s.split())
        try:
            ch, ri = cols.index("label_asym_id"), cols.index("label_seq_id")
        except ValueError:
            return 0
        for r in rows:
            if len(r) > max(ch, ri):
                seen.add((r[ch], r[ri]))
    else:                                          # PDB
        for line in text.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                seen.add((line[21], line[22:27]))
    return len(seen)


def _oom_of(text):
    """The LAST refusal in the log, not the first: the blocking paths retry past an early one."""
    for pat, kind in OOM_PATTERNS:
        ms = list(pat.finditer(text))
        if ms:
            m = ms[-1]
            g = {k: int(v) for k, v in m.groupdict().items() if v is not None}
            return {"class": kind, "mechanism": classify(g), "bytes": g,
                    "text": m.group(0)[:400].replace("\n", " ")}
    return None


def _tail(text: str, limit: int = 1600) -> str:
    """Bounded log excerpt that keeps BOTH ends, so the diagnosis survives wherever it sits.

    `text[-1200:]` kept only the trailing click frames. A device-open failure writes its fatal at
    the TOP of the log -- `_report_fatal` goes to the launcher's real stderr before anything else
    runs -- so the recorded tails for opendde-abag and protenix-v1 at 1536 contained no trace of
    the wedged card that caused them, and the rows could not be re-judged from what they stored.
    Same lesson `worker._err_text` already carries: when you do not know which end holds the
    payload, keep both.
    """
    text = (text or "").replace("\n", " | ")
    if len(text) <= limit:
        return text
    head = (limit - 5) // 2
    return text[:head] + " ... " + text[-(limit - 5 - head):]


def _write(row):
    OUTROOT.mkdir(parents=True, exist_ok=True)
    # report.py --backfill rewrites this file whole, so an unlocked append can be dropped
    # (gate-mandated-write-to-single-owner-file-is-a-race). Both sides take the same lock.
    with (OUTROOT / "results.jsonl").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(row) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    line = (f"{row['when']} {row['model']:14s} {row['size']:5d}tok {row['verdict']:8s} "
            f"wall={row['wall_s']:7.1f}s engine={row['engine_runtime_s']} "
            f"cif_res={row['cif_residues']}/{row['size']} ntok={row['n_tokens']} "
            f"rss={row['peak_host_rss_gib']}G "
            f"oom={row['oom']['class'] + '/' + row['oom']['mechanism'] if row['oom'] else '-'}"
            f"{'(fatal)' if row['fatal_oom'] else ''}"
            + (f" breaks={row['struct'].get('ca_breaks')} "
               f"worst_ca={row['struct'].get('worst_ca_ca')} "
               f"clash={row['struct'].get('clash_frac')}" if row.get("struct") else "")
            + (" ffn_fallback=YES" if row.get("ffn_fallback") else ""))
    with (OUTROOT / "sweep.log").open("a") as fh:
        fh.write(line + "\n")
    print(line)


def _group_alive(pgid: int) -> bool:
    """Is any NON-ZOMBIE process still in this group?

    `os.killpg(pgid, 0)` is the obvious check and it is wrong here: a signalled child stays a
    zombie until it is reaped, killpg succeeds on it, and the grace loop then spins the whole
    10 s on a group that is already dead. Measured: it turned a 3 s stall detection into a 28 s
    teardown, which eats the time the detector exists to save. Read the state out of
    /proc/<pid>/stat instead.
    """
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            # `comm` can contain spaces and parentheses, so split on the LAST ')'.
            fields = open(f"/proc/{entry.name}/stat").read().rpartition(")")[2].split()
        except OSError:
            continue
        if len(fields) < 3:
            continue
        state, pgrp = fields[0], fields[2]
        if pgrp == str(pgid) and state != "Z":
            return True
    return False


def kill_tree(proc, grace: float = 10.0) -> None:
    """Kill a fold and everything it spawned, and do not return while any of it lives.

    `proc.kill()` alone kills the `tt_bio.main` parent and leaves its spawned WORKER, which is
    the process that holds the chip and the card lease. That worker has PDEATHSIG armed, so in
    the normal case the kernel SIGTERMs it -- but a frozen one cannot take a signal it never
    reaches a check for, and its heartbeat backstop never fired either. Measured twice on this
    ladder: opendde's worker lived 44 more minutes at 108 % CPU holding card 0's lease with
    `"released": null`, and opendde-abag's lived 11 more minutes the same way, both after their
    parent was gone. Each one blocked the rungs behind it.

    So the fold gets its own process group (`start_new_session=True` at Popen) and the whole
    group is signalled here: SIGTERM, then SIGKILL to whatever ignored it, which is what
    actually clears a process wedged inside ttnn.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return
    # Refuse to signal our OWN group, whatever the caller did. Without this guard a proc that
    # was started WITHOUT start_new_session shares this process's group, and the killpg below
    # takes down the harness, the shell that launched it and anything else in that group. Which
    # is not hypothetical: it killed a pytest session and the ssh around it the first time this
    # function ran against a test subprocess created in the caller's group.
    if pgid in (os.getpgrp(), 0, 1):
        proc.kill()
        try:
            proc.wait(timeout=15)
        except Exception:
            pass
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        deadline = time.time() + (grace if sig == signal.SIGTERM else 15.0)
        while time.time() < deadline:
            if not _group_alive(pgid):
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                return
            time.sleep(0.5)
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def watch(proc, log: Path, t0: float, *, budget: float, stall: float,
          peak_rss: int = 0, floor_avail: int = 1 << 62, poll: float = 2.0):
    """Watch a running fold; return (killed, stalled, peak_rss, floor_avail).

    Two ways to stop it. `budget` is the wall-clock cap. `stall` is the one that matters more:
    a fold that stops WRITING has stopped folding, and waiting out the remaining budget on a
    frozen process is dead card time -- opendde at 1536 froze at `trunk 9/10` and burned the
    other 23 minutes of its 2700 s.

    Liveness is read off the LOG and deliberately not off CPU or RSS. That stall sat at 111 %
    CPU with two threads busy-polling and its RSS frozen to the byte, so "is it using the CPU"
    answers yes on a run that is going nowhere. Kernel compilation, every stage line and every
    warning land in this file, so growth is the honest signal and the generous default (900 s)
    leaves room for a slow legitimate phase.
    """
    last_size, last_grew = -1, time.time()
    while proc.poll() is None:
        time.sleep(poll)
        r, av = sample_host()
        peak_rss = max(peak_rss, r)
        floor_avail = min(floor_avail, av)
        try:
            size = log.stat().st_size
        except OSError:
            size = last_size
        if size != last_size:
            last_size, last_grew = size, time.time()
        if stall and time.time() - last_grew > stall:
            kill_tree(proc)
            return True, True, peak_rss, floor_avail
        if time.time() - t0 > budget:
            kill_tree(proc)
            return True, False, peak_rss, floor_avail
    proc.wait()
    return False, False, peak_rss, floor_avail


def _judge_affinity(a, out, text, wall, proc, killed, peak_rss, floor_avail):
    """nesso1 writes no structure. The artifact is the scalar, so read it and require a number."""
    scores = sorted(out.rglob("*_affinity.json"))
    value = None
    if scores:
        try:
            d = json.loads(scores[0].read_text())
            for k in ("affinity_pred_value", "affinity", "pred_value", "value"):
                if isinstance(d.get(k), (int, float)):
                    value = float(d[k]); break
            if value is None:
                for v in d.values():
                    if isinstance(v, (int, float)):
                        value = float(v); break
        except Exception:
            value = None
    oom = _oom_of(text)
    ok = value is not None
    verdict = ("PASS" if ok else not_a_measurement(proc.returncode, text)
               or ("STALLED" if stalled else "TIMEOUT" if killed
                   else "OOM" if oom else "FAIL"))
    row = {"model": a.model, "size": a.size, "tag": a.tag, "task": "affinity",
           "verdict": verdict, "wall_s": round(wall, 1), "engine_runtime_s": None,
           "cif": str(scores[0]) if scores else None,
           "cif_residues": a.size if ok else 0, "n_tokens": None,
           "affinity": value, "exit": proc.returncode, "killed": killed,
           "single_sequence": True, "sampling_steps": None,
           "recycling_steps": a.recycling_steps,
           "peak_host_rss_gib": round(peak_rss / 2**30, 2),
           "floor_memavail_gib": round(floor_avail / 2**30, 2),
           "oom": oom, "fatal_oom": bool(oom and not ok),
           "tail": _tail(text) if verdict != "PASS" else "",
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(row)
    if verdict in ("CONTENDED", "WEDGED"):
        return CONTENDED_EXIT_CODE
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--budget", type=int, default=2400, help="seconds before the rung is killed")
    ap.add_argument("--contention_retries", type=int, default=2,
                    help="re-attempts when a co-tenant holds the card (verdict CONTENDED, "
                         "never FAIL: nothing ran, so nothing was measured)")
    ap.add_argument("--contention_wait", type=int, default=180,
                    help="seconds between contention re-attempts")
    ap.add_argument("--stall", type=int, default=900,
                    help="seconds of NO growth in fold.log before the rung is called STALLED. "
                         "opendde at 1536 froze at `trunk 9/10` and burned the remaining 23 min "
                         "of its 2700 s budget without writing another byte. 0 disables.")
    ap.add_argument("--debug", action="store_true",
                    help="pass --debug to predict, so the worker keeps stdout/stderr and any "
                         "engine line (e.g. the pair-FFN fallback) reaches fold.log")
    ap.add_argument("--sampling_steps", type=int, default=20)
    ap.add_argument("--recycling_steps", type=int, default=None)
    ap.add_argument("--single_sequence", action="store_true", default=True)
    ap.add_argument("--msa", dest="single_sequence", action="store_false")
    ap.add_argument("--tag", default="")
    ap.add_argument("--task", choices=("predict", "affinity"), default=None,
                    help="affinity scores a ligand and writes no structure, so it is judged on "
                         "its scalar rather than on a CIF. Derived from the model when omitted")
    a = ap.parse_args()

    # Which CLI verb this model takes, read off the registry rather than passed in by every
    # caller. `chain.sh` does not know one model from another and passed none, so nesso1 -- the
    # roster's only affinity model -- would have been handed to `predict`, whose --model choices
    # do not include it: a click usage error recorded as a FAILED 1536 rung for a model that was
    # never invoked. Same class as the CONTENDED and WEDGED rows, one layer earlier.
    if a.task is None:
        a.task = "affinity" if a.model in AFFINITY_MODELS else "predict"

    fixture = (WT / "perf" / "bh1536" / "fixtures" / f"aff_{a.size}.yaml" if a.task == "affinity"
               else WT / "perf" / "size512" / "fixtures" / f"cdk2x2_{a.size}.yaml")
    if not fixture.is_file():
        sys.exit(f"no fixture {fixture}")
    label = f"{a.model}_{a.size}" + (f"_{a.tag}" if a.tag else "")
    out = OUTROOT / "runs" / label
    if out.exists():
        shutil.rmtree(out)          # a stale results folder has been read as this run's before
    out.mkdir(parents=True)
    log = out / "fold.log"

    cmd = [PY, "-m", "tt_bio.main", a.task, str(fixture), "--model", a.model,
           "--out_dir", str(out), "--accelerator", "tenstorrent"]
    if a.task == "predict":
        cmd += ["--sampling_steps", str(a.sampling_steps), "--diffusion_samples", "1",
                "--seed", "0", "--output_format", "cif"]
        if a.single_sequence:
            cmd.append("--single_sequence")
    if a.recycling_steps is not None:
        cmd += ["--recycling_steps", str(a.recycling_steps)]
    if a.debug:
        cmd.append("--debug")

    env = dict(os.environ)
    env.update(PYTHONPATH=str(WT), TT_VISIBLE_DEVICES="0", TT_BIO_LEASE_CARDS="0",
               TT_BIO_LEASE_HOLDER="worker:bh-1536-structure")

    # A co-tenant with the card is transient, so wait it out rather than burning the rung:
    # the sibling pytest that took card 0 tonight holds it for a test, not for the night.
    # Bounded, and the last attempt's log is what gets judged either way.
    t0 = time.time()
    peak_rss, floor_avail = 0, 1 << 62
    for attempt in range(a.contention_retries + 1):
        # --budget is the FOLD's budget, so it restarts with each attempt. Measuring it from t0
        # made a lease wait and its 180 s backoff eat the fold: opendde:1536 retried once and
        # then folded on roughly 2230 of its nominal 2700 s. A rung recorded TIMEOUT because it
        # started late is a ceiling that says more about the co-tenant than the silicon. `wall`
        # below stays the honest total, retries and all.
        t_attempt = time.time()
        with log.open("wb") as fh:
            # Own process group, so kill_tree can take the fold AND its spawned worker as a
            # unit. The finally is the backstop: if this harness dies for any other reason,
            # it must not leave a worker holding the card behind it.
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=str(WT), env=env, start_new_session=True)
            try:
                killed, stalled, peak_rss, floor_avail = watch(
                    proc, log, t_attempt, budget=a.budget, stall=a.stall,
                    peak_rss=peak_rss, floor_avail=floor_avail)
            finally:
                if proc.poll() is None:
                    kill_tree(proc)
        text = log.read_text(errors="replace")
        if not contended(proc.returncode, text) or killed or attempt == a.contention_retries:
            break
        print(f"card 0 held by a co-tenant, retrying {a.model} {a.size} in "
              f"{a.contention_wait}s (attempt {attempt + 2}/{a.contention_retries + 1})",
              flush=True)
        time.sleep(a.contention_wait)
    wall = time.time() - t0
    fold_wall = time.time() - t_attempt   # the judged attempt only, without any lease waiting
    no_measure = not_a_measurement(proc.returncode, text)

    # --- judge the artifact, never the exit code -------------------------------------------
    if a.task == "affinity":
        return _judge_affinity(a, out, text, wall, proc, killed, peak_rss, floor_avail)
    cifs = sorted(out.rglob("structures/*.cif")) + sorted(out.rglob("structures/*.pdb"))
    res_json = sorted(out.rglob("results.json"))
    nres = cif_residues(cifs[0]) if cifs else 0
    metrics = {}
    if res_json:
        try:
            metrics = json.loads(res_json[0].read_text())
        except Exception:
            metrics = {}
    if isinstance(metrics, list):          # boltz2 writes a list, one entry per target
        metrics = metrics[0] if metrics else {}
    ntok = None
    for k in ("n_tokens", "num_tokens", "tokens"):
        v = metrics.get(k) if isinstance(metrics, dict) else None
        if isinstance(v, int):
            ntok = v; break
    runtime_s = metrics.get("runtime_s") if isinstance(metrics, dict) else None
    plddt = next((metrics[k] for k in ("complex_plddt", "plddt", "mean_plddt")
                  if isinstance(metrics, dict) and isinstance(metrics.get(k), (int, float))), None)

    oom = _oom_of(text)
    # An L1 refusal that the blocking path retried past is not a failure; only count one
    # that the run did not survive.
    fatal_oom = oom if (oom and nres == 0) else None

    # Did the release-gated pair-FFN fallback serve this fold? A PASS that needed it is not a
    # shipped-engine PASS, and only the engine's own line can say so: fragmentation is stateful,
    # so a re-run of a rung that OOM'd once can pass on a tidier chip with nothing fixed.
    # Only visible with --debug, hence None (not False) when the run could not have reported it.
    ffn_fallback = ("[pair-ffn] DRAM refused" in text) if a.debug else None

    # A CIF that exists is not a fold. `perf/ceilings/struct_signal.py` is the repo's one
    # structural instrument (it imports the release gate's own thresholds), so a rung that
    # returns coordinates still has to show a continuous backbone before it counts as PASS.
    signal = {}
    if cifs:
        try:
            sig = subprocess.run([PY, str(WT / "perf/ceilings/struct_signal.py"), str(out)],
                                 capture_output=True, text=True, cwd=str(WT), env=env,
                                 timeout=1800)
            signal = json.loads(sig.stdout.strip().splitlines()[-1])
        except Exception as exc:
            signal = {"scored": 0, "error": str(exc)[:200]}

    ok = bool(cifs) and nres == a.size
    # CONTENDED/WEDGED outrank OOM and FAIL: a run that never got a working card cannot have
    # found a wall. STALLED and TIMEOUT outrank OOM only because a surviving refusal that the
    # run retried past is not what ended it.
    verdict = ("PASS" if ok else no_measure
               or ("STALLED" if stalled else "TIMEOUT" if killed
                   else "OOM" if fatal_oom else "FAIL"))
    row = {"model": a.model, "size": a.size, "tag": a.tag, "task": "predict",
           "verdict": verdict, "wall_s": round(wall, 1), "engine_runtime_s": runtime_s,
           "cif": str(cifs[0]) if cifs else None, "cif_residues": nres,
           "n_tokens": ntok, "plddt": plddt, "struct": signal,
           "exit": proc.returncode, "killed": killed,
           "single_sequence": a.single_sequence, "sampling_steps": a.sampling_steps,
           "recycling_steps": a.recycling_steps,
           "peak_host_rss_gib": round(peak_rss / 2**30, 2),
           "floor_memavail_gib": round(floor_avail / 2**30, 2),
           "oom": oom, "fatal_oom": bool(fatal_oom), "ffn_fallback": ffn_fallback,
           "debug": a.debug, "stalled": stalled, "fold_wall_s": round(fold_wall, 1),
           "attempts": attempt + 1,
           # Where it stopped is the diagnosis for a stall, so keep the last line it wrote.
           "last_progress": (next((l for l in reversed(text.splitlines()) if l.strip()), "")
                             if stalled else None),
           "tail": _tail(text) if verdict != "PASS" else "",
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(row)
    # 75 (EX_TEMPFAIL, the same code tt_bio uses for a contended card) when this rung measured
    # NOTHING, so a caller can tell "the model failed" from "the card was unusable" by return
    # code. chain.sh counts these and stops rather than feeding a dead card its whole queue.
    if verdict in ("CONTENDED", "WEDGED"):
        return CONTENDED_EXIT_CODE
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
