"""One BoltzGen design rung, run to completion, with the whole log kept and the SDPA route
counters read out of the fold's own process.

`perf/bhdesign/ladder.py` keeps a 2500-character tail of a failed rung and nothing at all of a
rung that passes, which is the right trade for a 30-rung sweep and the wrong one here: the
question is what the triangle-attention SDPA picked at 2208 padded tokens and how long the fold
took, and both live in the middle of the log. So this keeps the full log on disk and installs
`perf/fused_sdpa/route_census.py`'s atexit hook in every process of the fold, then verifies the
artifact with the ladder's own checker so the verdict is the same verdict.

    python3 perf/bgsdpa/rung.py --target-res 2100 --timeout 5400 --card 2
"""
import argparse
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stop(proc, grace):
    """SIGINT the fold's process group, give atexit `grace` seconds, then SIGKILL."""
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            return proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    return proc.poll() if proc.poll() is not None else -9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-res", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--card", default="2")
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--holder", default="worker:bh-boltzgen-sdpa-circbuf")
    ap.add_argument("--target", default="perf/bhdesign/targets/big_7324.cif")
    ap.add_argument("--work", type=pathlib.Path, default=ROOT / "perf" / "bgsdpa" / "work")
    ap.add_argument("--tag", default="")
    ap.add_argument("--env", action="append", default=[], help="K=V for the fold's process")
    ap.add_argument("--grace", type=int, default=120,
                    help="seconds after SIGINT for the fold's atexit hooks to write the census")
    args = ap.parse_args()

    ladder = _load("bh_ladder", ROOT / "perf" / "bhdesign" / "ladder.py")
    census = _load("route_census", ROOT / "perf" / "fused_sdpa" / "route_census.py")

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"r{args.target_res}"
    fx, atoms, tres = ladder.boltzgen_fixture(work, args.target_res,
                                              pathlib.Path(args.target), args.binder)
    out_dir = work / f"out_{tag}"
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)

    # The census hook, in a sitecustomize this run owns. Appended to sys.path, never prepended:
    # nothing here may shadow a stdlib module for the fold.
    hook = work / f"hook_{tag}"
    hook.mkdir(exist_ok=True)
    dumps = work / f"dumps_{tag}"
    dumps.mkdir(exist_ok=True)
    for stale in dumps.glob("pid*.json"):
        stale.unlink()
    (hook / "sitecustomize.py").write_text(
        "import sys\n"
        f"sys.path.append({str(ROOT)!r})\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('_rc', {str(ROOT / 'perf' / 'fused_sdpa' / 'route_census.py')!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "m.install_hook()\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{hook}:{ROOT}"
    env["ROUTE_CENSUS_DIR"] = str(dumps)
    env["TT_VISIBLE_DEVICES"] = args.card
    env["TT_BIO_LEASE_CARDS"] = args.card
    env["TT_BIO_LEASE_HOLDER"] = args.holder
    env["TT_BIO_SIZE_LIMIT"] = "0"      # this IS the ceiling measurement; it cannot obey one
    env["TT_BIO_LEASE_TIMEOUT"] = str(args.timeout)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    cmd = [ladder.PY, "-u", "-m", "tt_bio.main", "design", str(fx), "--model", "boltzgen",
           "--out_dir", str(out_dir), "--num_designs", "1", "--steps", "design", "--debug"]
    log = work / f"log_{tag}.txt"
    t0 = time.time()
    with log.open("w") as fh:
        fh.write(f"# {' '.join(cmd)}\n# target_res={tres} target_atoms={atoms} "
                 f"card={args.card} timeout={args.timeout}s env={args.env}\n")
        fh.flush()
        # SIGINT first, not SIGKILL. subprocess.run(timeout=) kills with SIGKILL, which skips
        # every atexit hook -- including the route census this whole script exists to collect --
        # and leaves the chip dirty for the next opener. SIGINT raises KeyboardInterrupt in the
        # fold, atexit runs, the census lands, and the device closes itself. The group is the
        # fold's own (start_new_session), so this signals the spawned design workers too and
        # nothing outside this rung.
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = _stop(proc, args.grace)
    wall = round(time.time() - t0, 1)

    ok, detail = ladder.check_artifact(("designcif", (args.binder, tres)), out_dir, "boltzgen")
    agg: dict = {}
    for p in sorted(dumps.glob("pid*.json")):
        census._merge(agg, json.loads(p.read_text()))
    rec = {"target_res": tres, "target_atoms": atoms, "rc": rc, "timed_out": timed_out,
           "wall_s": wall, "verdict": "PASS" if (ok and rc == 0) else "FAIL",
           "artifact": detail, "log": str(log), "census": agg, "env": args.env,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = work / f"rung_{tag}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "census"}, indent=2))
    print("census:", json.dumps(agg, indent=2))


main()
